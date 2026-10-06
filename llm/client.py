"""
Client gọi LMDeploy Server qua API tương thích OpenAI.

LMDeploy tự chạy continuous batching trên GPU, nên phía service chỉ cần gửi
các request song song; không cần tự gom batch ở tầng này.

Module cung cấp hai hàm chính:
- ``complete``: sinh xong mới trả về (non-streaming).
- ``stream``: sinh dần từng token (streaming).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional

from openai import AsyncOpenAI

import settings

logger = logging.getLogger(__name__)

_client: Optional[AsyncOpenAI] = None


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
