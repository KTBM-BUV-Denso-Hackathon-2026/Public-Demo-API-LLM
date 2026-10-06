"""
Entrypoint FastAPI của microservice ``llm_api``.

Endpoint:
- ``GET  /health``      : kiểm tra sống/chết.
- ``POST /chat``        : một lượt hỏi - đáp (JSON hoặc SSE nếu ``stream=true``).
- ``POST /chat/batch``  : nhiều lượt hỏi - đáp (JSON hoặc SSE gộp nếu ``stream=true``).

Chạy:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

import services
from schemas import (
    BatchChatRequest,
    BatchChatResponse,
    ChatRequest,
    ChatResponse,
    StreamEvent,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DENSO LLM API",
    version="1.0.0",
    description=(
        "Microservice RAG + LLM. Hỗ trợ batch inference, RAG search và "
        "streaming (SSE) tới LMDeploy Server."
    ),
)

SSE_MEDIA_TYPE = "text/event-stream"
# Ngăn proxy (nginx...) đệm response, đảm bảo token được đẩy đi ngay.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------
def _to_sse(event: StreamEvent) -> str:
    """
    Mã hoá một ``StreamEvent`` thành khung SSE.

    Dùng ``exclude_none`` để payload gọn, và ``ensure_ascii=False`` để giữ
    nguyên tiếng Việt trong JSON.
    """
    payload = event.model_dump(exclude_none=True)
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _sse_generator(
    events: AsyncIterator[StreamEvent],
) -> AsyncIterator[str]:
    """Bọc một luồng ``StreamEvent`` thành luồng text đúng định dạng SSE."""
    async for event in events:
        yield _to_sse(event)


def _streaming_response(events: AsyncIterator[StreamEvent]) -> StreamingResponse:
    """Tạo ``StreamingResponse`` SSE từ luồng event."""
    return StreamingResponse(
        _sse_generator(events),
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_HEADERS,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", tags=["system"])
async def health() -> dict:
    """Endpoint kiểm tra service còn sống."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest):
    """
    Xử lý một lượt hỏi - đáp.

    - ``stream=false`` (mặc định): trả JSON ``ChatResponse``.
    - ``stream=true``: trả SSE, mỗi event là một ``StreamEvent``.

    RAG search luôn chạy trước khi gọi model, kể cả ở chế độ streaming, nên
    câu trả lời vẫn dựa trên context truy hồi được.
    """
    if request.stream:
        return _streaming_response(services.stream_single(request))

    try:
        return await services.answer(request)
    except Exception as exc:  # noqa: BLE001 - trả lỗi rõ ràng cho client
        logger.exception("Lỗi xử lý /chat cho chat_id=%s", request.chat_id)
        raise HTTPException(
            status_code=502,
            detail=f"Lỗi khi xử lý yêu cầu: {type(exc).__name__}: {exc}",
        ) from exc


@app.post("/chat/batch", response_model=BatchChatResponse, tags=["chat"])
async def chat_batch(request: BatchChatRequest):
    """
    Xử lý nhiều lượt hỏi - đáp trong một request.

    - ``stream=false`` (mặc định): trả JSON ``BatchChatResponse`` (mỗi item
      một kết quả, đúng thứ tự).
    - ``stream=true``: trả một SSE stream gộp; mỗi event mang ``index`` để
      client tách luồng theo item.

    Khi không streaming, các item được chạy song song (``asyncio.gather``)
    và LMDeploy sẽ tự gom batch trên GPU.
    """
    if request.stream:
        return _streaming_response(services.stream_batch(request.items))

    try:
        results = await services.answer_batch(request.items)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Lỗi xử lý /chat/batch (%d item)", len(request.items))
        raise HTTPException(
            status_code=502,
            detail=f"Lỗi khi xử lý batch: {type(exc).__name__}: {exc}",
        ) from exc

    return BatchChatResponse(results=results)
