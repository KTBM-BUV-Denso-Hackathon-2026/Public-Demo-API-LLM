"""
Tầng nghiệp vụ (service layer) của ``llm_api``.

Chịu trách nhiệm ghép ba bước: truy hồi RAG -> dựng prompt -> gọi LLM.
Hỗ trợ 3 chế độ:

- **single**: một câu hỏi, trả về một lần.
- **batch**: nhiều câu hỏi, chạy song song, trả về một lần.
- **stream**: trả dần từng token qua SSE (single hoặc batch).

Điểm tiêm phụ thuộc (dependency injection) để test:
- ``retrieve_contexts``: có thể monkeypatch để bỏ qua vector DB.
- ``_complete`` / ``_stream``: có thể monkeypatch để bỏ qua LMDeploy.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, List, Optional, Sequence, Tuple, Union

import settings
from llm import client as llm_client
from llm.client import GenerationParams
from schemas import ChatMessage, ChatRequest, ChatResponse, StreamEvent

logger = logging.getLogger(__name__)

# Sentinel đánh dấu một worker đã hoàn tất (dùng khi gộp stream của batch).
_DONE = object()


# ---------------------------------------------------------------------------
# Dựng prompt
# ---------------------------------------------------------------------------
def build_rag_prompt(context: str, question: str) -> str:
    """
    Ghép context RAG với câu hỏi gốc thành nội dung user message.

    Nếu không có context (RAG tắt hoặc không tìm thấy chunk), trả về đúng
    câu hỏi gốc để model không bị nhiễu bởi phần "THÔNG TIN THAM KHẢO" rỗng.
    """
    if not context:
        return question
    return f"THÔNG TIN THAM KHẢO:\n{context}\n\nCÂU HỎI: {question}"


def build_messages(
    history: Sequence[ChatMessage],
    context: str,
    question: str,
) -> List[dict]:
    """
    Dựng danh sách messages gửi cho LLM.

    Context RAG chỉ được chèn vào lượt hỏi MỚI NHẤT; history giữ nguyên bản
    gốc (không nhét lại context cũ) để tiết kiệm token và tránh lặp.
    """
    messages = [{"role": message.role, "content": message.content} for message in history]
    messages.append({"role": "user", "content": build_rag_prompt(context, question)})
    return messages


def build_updated_history(request: ChatRequest, answer: str) -> List[ChatMessage]:
    """
    Trả về history đã cập nhật sau khi sinh xong.

    Chỉ lưu câu hỏi GỐC (không kèm context RAG) và câu trả lời của AI.
    Caller chịu trách nhiệm persist danh sách này (ví dụ vào ``Chat.message``).
    """
    messages = list(request.history)
    messages.append(ChatMessage(role="user", content=request.new_question))
    messages.append(ChatMessage(role="assistant", content=answer))
    return messages


def resolve_params(request: ChatRequest) -> GenerationParams:
    """Gộp tham số sinh của request với giá trị mặc định trong settings."""
    return GenerationParams(
        model=settings.LLM_MODEL,
        temperature=(
            request.temperature
            if request.temperature is not None
            else settings.DEFAULT_TEMPERATURE
        ),
        top_p=request.top_p if request.top_p is not None else settings.DEFAULT_TOP_P,
        max_new_tokens=(
            request.max_new_tokens
            if request.max_new_tokens is not None
            else settings.DEFAULT_MAX_NEW_TOKENS
        ),
    )


def resolve_top_k(request: ChatRequest) -> int:
    """Số chunk RAG cho request; mặc định lấy ``settings.RAG_TOP_K``."""
    return request.top_k if request.top_k is not None else settings.RAG_TOP_K


# ---------------------------------------------------------------------------
# Truy hồi RAG (async)
# ---------------------------------------------------------------------------
async def retrieve_contexts(questions: Sequence[str], top_k: int) -> List[str]:
    """
    Truy hồi context cho nhiều câu hỏi (batch encode + batch search).

    Chạy phần đồng bộ (encode + psycopg2) trong thread riêng để không chặn
    event loop. Nếu ``settings.RAG_ENABLED`` là False, trả về toàn chuỗi rỗng.

    Args:
        questions: Danh sách câu hỏi.
        top_k: Số chunk cho mỗi câu hỏi.

    Returns:
        Danh sách context, giữ đúng thứ tự với ``questions``.

    Raises:
        Exception: Lỗi từ tầng DB/embedding được ném nguyên vẹn để caller
            biết và xử lý (fail-fast, không âm thầm trả context rỗng).
    """
    questions = list(questions)
    if not questions:
        return []
    if not settings.RAG_ENABLED:
        return ["" for _ in questions]

    from database import controllers

    return await asyncio.to_thread(controllers.batch_search_rag_data, questions, top_k)


async def retrieve_contexts_varied(
    questions: Sequence[str],
    top_ks: Sequence[int],
) -> List[str]:
    """
    Truy hồi context khi mỗi câu hỏi có ``top_k`` khác nhau.

    Gom các câu hỏi theo cùng ``top_k`` để vẫn tận dụng batch encode, sau đó
    trả kết quả về đúng vị trí ban đầu.
    """
    questions = list(questions)
    top_ks = list(top_ks)
    if len(questions) != len(top_ks):
        raise ValueError("questions và top_ks phải cùng độ dài.")

    # Nhóm các index theo giá trị top_k.
    groups: dict = {}
    for index, top_k in enumerate(top_ks):
        groups.setdefault(top_k, []).append(index)

    results: List[str] = ["" for _ in questions]
    if not groups:
        return results

    group_items = list(groups.items())
    grouped_contexts = await asyncio.gather(
        *[
            retrieve_contexts([questions[i] for i in indexes], top_k)
            for top_k, indexes in group_items
        ]
    )
    for (top_k, indexes), contexts in zip(group_items, grouped_contexts):
        for index, context in zip(indexes, contexts):
            results[index] = context
    return results


# ---------------------------------------------------------------------------
# Điểm tiêm LLM (bọc mỏng quanh llm.client để dễ monkeypatch khi test)
# ---------------------------------------------------------------------------
async def _complete(messages: List[dict], params: GenerationParams) -> str:
    """Gọi LLM non-streaming."""
    return await llm_client.complete(messages, params)


async def _stream(messages: List[dict], params: GenerationParams) -> AsyncIterator[str]:
    """Gọi LLM streaming, yield từng delta."""
    async for delta in llm_client.stream(messages, params):
        yield delta


# ---------------------------------------------------------------------------
# Sinh câu trả lời
# ---------------------------------------------------------------------------
async def _answer_with_context(request: ChatRequest, context: str) -> ChatResponse:
    """Sinh câu trả lời cho một request khi đã có sẵn context RAG."""
    messages = build_messages(request.history, context, request.new_question)
    answer = await _complete(messages, resolve_params(request))
    return ChatResponse(
        chat_id=request.chat_id,
        answer=answer,
        messages=build_updated_history(request, answer),
        contexts=[context] if context else [],
    )


async def answer(request: ChatRequest) -> ChatResponse:
    """
    Xử lý một lượt hỏi - đáp đơn lẻ (non-streaming).

    Args:
        request: Yêu cầu từ client.

    Returns:
        ``ChatResponse`` chứa câu trả lời và history đã cập nhật.
    """
    contexts = await retrieve_contexts([request.new_question], resolve_top_k(request))
    return await _answer_with_context(request, contexts[0] if contexts else "")


async def answer_batch(requests: Sequence[ChatRequest]) -> List[ChatResponse]:
    """
    Xử lý NHIỀU lượt hỏi - đáp song song (non-streaming batch inference).

    Context RAG được truy hồi một lần cho cả batch (batch encode), sau đó các
    request được gửi song song tới LMDeploy. LMDeploy tự gom chúng thành một
    batch trên GPU (continuous batching).

    Args:
        requests: Danh sách yêu cầu.

    Returns:
        Danh sách ``ChatResponse`` giữ đúng thứ tự đầu vào.
    """
    requests = list(requests)
    if not requests:
        return []

    contexts = await retrieve_contexts_varied(
        [request.new_question for request in requests],
        [resolve_top_k(request) for request in requests],
    )
    return await asyncio.gather(
        *[
            _answer_with_context(request, context)
            for request, context in zip(requests, contexts)
        ]
    )


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
async def stream_single(
    request: ChatRequest,
    index: Optional[int] = None,
    context: Optional[str] = None,
) -> AsyncIterator[StreamEvent]:
    """
    Sinh SSE event cho một request.

    Args:
        request: Yêu cầu từ client.
        index: Vị trí item trong batch (None nếu request đơn lẻ).
        context: Context RAG đã tính trước; nếu None sẽ tự truy hồi.

    Yields:
        Chuỗi ``StreamEvent`` theo trình tự ``start`` -> nhiều ``delta`` ->
        ``end`` (hoặc ``error`` nếu có lỗi).
    """
    try:
        if context is None:
            contexts = await retrieve_contexts(
                [request.new_question], resolve_top_k(request)
            )
            context = contexts[0] if contexts else ""

        messages = build_messages(request.history, context, request.new_question)
        params = resolve_params(request)

        yield StreamEvent(type="start", chat_id=request.chat_id, index=index)

        parts: List[str] = []
        async for delta in _stream(messages, params):
            parts.append(delta)
            yield StreamEvent(
                type="delta", chat_id=request.chat_id, index=index, delta=delta
            )

        yield StreamEvent(
            type="end",
            chat_id=request.chat_id,
            index=index,
            messages=build_updated_history(request, "".join(parts)),
        )
    except Exception as exc:  # noqa: BLE001 - trả lỗi về client qua SSE
        logger.exception("Lỗi khi streaming chat_id=%s", request.chat_id)
        yield StreamEvent(
            type="error",
            chat_id=request.chat_id,
            index=index,
            error=f"{type(exc).__name__}: {exc}",
        )


async def stream_batch(requests: Sequence[ChatRequest]) -> AsyncIterator[StreamEvent]:
    """
    Gộp stream của nhiều request thành MỘT stream SSE duy nhất.

    Mỗi item chạy trong một task riêng và đẩy event vào chung một queue; event
    mang thêm ``index`` để client tách luồng. Cách này cho phép các câu trả
    lời xen kẽ nhau nhưng vẫn giữ đúng thứ tự trong từng item.

    Args:
        requests: Danh sách yêu cầu.

    Yields:
        ``StreamEvent`` của mọi item, có ``index`` định danh.
    """
    requests = list(requests)
    if not requests:
        return

    # Truy hồi RAG một lần cho cả batch trước khi bắt đầu stream.
    try:
        contexts = await retrieve_contexts_varied(
            [request.new_question for request in requests],
            [resolve_top_k(request) for request in requests],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Lỗi truy hồi RAG cho batch streaming")
        for index, request in enumerate(requests):
            yield StreamEvent(
                type="error",
                chat_id=request.chat_id,
                index=index,
                error=f"{type(exc).__name__}: {exc}",
            )
        return

    queue: "asyncio.Queue[Union[StreamEvent, object]]" = asyncio.Queue()

    async def worker(index: int, request: ChatRequest, context: str) -> None:
        """Chạy một item và đẩy toàn bộ event của nó vào queue chung."""
        try:
            async for event in stream_single(request, index=index, context=context):
                await queue.put(event)
        finally:
            await queue.put(_DONE)

    tasks = [
        asyncio.create_task(worker(index, request, context))
        for index, (request, context) in enumerate(zip(requests, contexts))
    ]

    try:
        remaining = len(tasks)
        while remaining > 0:
            item = await queue.get()
            if item is _DONE:
                remaining -= 1
                continue
            yield item  # type: ignore[misc]
    finally:
        # Huỷ các task còn lại nếu client ngắt kết nối giữa chừng.
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
