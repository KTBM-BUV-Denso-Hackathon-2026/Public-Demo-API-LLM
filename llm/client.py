"""
Client gọi model, hỗ trợ hai provider (``settings.LLM_PROVIDER``):

- ``openai_compatible`` (mặc định): gọi LMDeploy Server qua API tương thích
  OpenAI. LMDeploy tự continuous-batching trên GPU nên phía service chỉ cần gửi
  các request song song.
- ``mock``: sinh câu trả lời giả lập tại chỗ, KHÔNG cần GPU/model. Chỉ dùng cho
  dev/test — nội dung trả về được đánh dấu rõ ``[MOCK]`` để không lẫn với kết
  quả thật.

Module cung cấp hai hàm chính:
- ``complete``: sinh xong mới trả về (non-streaming).
- ``stream``: sinh dần từng đoạn text (streaming).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional

from openai import AsyncOpenAI

import settings

logger = logging.getLogger(__name__)

_client: Optional[AsyncOpenAI] = None

#: Nhãn đánh dấu câu trả lời của provider ``mock``.
MOCK_MARKER = "[MOCK]"


@dataclass
class GenerationParams:
    """
    Tham số sinh dùng chung cho cả hai chế độ.

    Giá trị mặc định lấy từ ``settings`` (có thể override theo từng request).
    """

    model: str = settings.LLM_MODEL
    temperature: float = settings.DEFAULT_TEMPERATURE
    top_p: float = settings.DEFAULT_TOP_P
    max_new_tokens: int = settings.DEFAULT_MAX_NEW_TOKENS


def get_client() -> AsyncOpenAI:
    """
    Trả về ``AsyncOpenAI`` client dùng chung (singleton).

    Client được tạo một lần và tái sử dụng cho mọi request, giúp tái dùng
    connection pool thay vì tạo mới liên tục.
    """
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.LM_DEPLOY_API_KEY,
            base_url=settings.LM_DEPLOY_BASE_URL,
            timeout=settings.LLM_REQUEST_TIMEOUT,
        )
    return _client


def is_mock() -> bool:
    """True nếu đang dùng provider giả lập."""
    return settings.LLM_PROVIDER == "mock"


# ---------------------------------------------------------------------------
# Provider giả lập (dev/test)
# ---------------------------------------------------------------------------
def _mock_answer(messages: List[dict]) -> str:
    """
    Dựng câu trả lời giả lập nhưng phản ánh đúng dữ liệu đầu vào.

    Hàm bóc phần context RAG (nếu có) để chứng minh context đã được truyền
    xuống đúng, giúp kiểm thử đường ống RAG mà không cần model thật.
    """
    question = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            question = message.get("content", "")
            break

    context_block = ""
    if "THÔNG TIN THAM KHẢO:" in question:
        context_block = question.split("THÔNG TIN THAM KHẢO:", 1)[1]
        context_block = context_block.split("CÂU HỎI:", 1)[0].strip()

    question_only = question
    if "CÂU HỎI:" in question:
        question_only = question.split("CÂU HỎI:", 1)[1].strip()

    chunks = [
        line[2:].strip()
        for line in context_block.splitlines()
        if line.strip().startswith("- ")
    ]

    lines = [f"{MOCK_MARKER} Đây là câu trả lời sinh bởi provider mock (không có model thật)."]
    lines.append(f"Câu hỏi nhận được: {question_only}")
    if chunks:
        lines.append(f"Số đoạn ngữ cảnh được truy hồi: {len(chunks)}")
        preview = chunks[0][:160]
        lines.append(f"Đoạn ngữ cảnh đầu tiên: {preview}")
    else:
        lines.append("Không có ngữ cảnh RAG nào được truy hồi.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# API công khai
# ---------------------------------------------------------------------------
async def complete(
    messages: List[dict],
    params: Optional[GenerationParams] = None,
) -> str:
    """
    Sinh câu trả lời ở chế độ non-streaming.

    Args:
        messages: Danh sách tin nhắn theo chuẩn OpenAI (role/content).
        params: Tham số sinh; mặc định dùng giá trị trong settings.

    Returns:
        Nội dung câu trả lời của model (chuỗi, có thể rỗng).
    """
    params = params or GenerationParams()
    if is_mock():
        return _mock_answer(messages)

    response = await get_client().chat.completions.create(
        model=params.model,
        messages=messages,
        temperature=params.temperature,
        top_p=params.top_p,
        max_tokens=params.max_new_tokens,
    )
    return response.choices[0].message.content or ""


async def stream(
    messages: List[dict],
    params: Optional[GenerationParams] = None,
) -> AsyncIterator[str]:
    """
    Sinh câu trả lời ở chế độ streaming, yield từng đoạn text (delta).

    Args:
        messages: Danh sách tin nhắn theo chuẩn OpenAI.
        params: Tham số sinh; mặc định dùng giá trị trong settings.

    Yields:
        Từng đoạn content mới sinh (không rỗng).
    """
    params = params or GenerationParams()

    if is_mock():
        delay = max(settings.MOCK_STREAM_DELAY_MS, 0) / 1000.0
        for token in _mock_answer(messages).split(" "):
            if delay:
                await asyncio.sleep(delay)
            yield token + " "
        return

    response_stream = await get_client().chat.completions.create(
        model=params.model,
        messages=messages,
        temperature=params.temperature,
        top_p=params.top_p,
        max_tokens=params.max_new_tokens,
        stream=True,
    )

    async for chunk in response_stream:
        # Một số chunk (ví dụ chunk chứa usage) không có choices.
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            yield content
