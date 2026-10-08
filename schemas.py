"""
Pydantic schemas cho API của ``llm_api``.

Ba nhóm chính:
- ``ChatRequest`` / ``BatchChatRequest``: đầu vào.
- ``ChatResponse`` / ``BatchChatResponse``: đầu ra khi ``stream=False``.
- ``StreamEvent``: payload của từng sự kiện khi ``stream=True`` (SSE).
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

import settings


class ChatMessage(BaseModel):
    """Một lượt tin nhắn trong hội thoại (tương thích chuẩn OpenAI)."""

    role: Literal["system", "user", "assistant"] = Field(
        ..., description="Vai trò của tin nhắn."
    )
    content: str = Field(
        default="",
        max_length=settings.MAX_MESSAGE_LENGTH,
        description="Nội dung tin nhắn.",
    )


class ChatRequest(BaseModel):
    """
    Đầu vào cho một lượt hỏi - đáp đơn lẻ.

    Attributes:
        chat_id: ID cuộc hội thoại (tương ứng ``Chat.id`` ở denso_backend).
            Dùng để định danh hội thoại thay cho ``user_id`` trước đây.
        new_question: Câu hỏi mới của người dùng (KHÔNG kèm RAG context).
        history: Các tin nhắn trước đó (không gồm câu hỏi mới). Caller chịu
            trách nhiệm persist, ``llm_api`` chỉ đọc.
        stream: Nếu True, trả về SSE streaming thay vì JSON một lần.
        top_k: Số chunk RAG truy hồi; mặc định lấy ``settings.RAG_TOP_K``.
        temperature / top_p / max_new_tokens: override tham số sinh.
    """

    chat_id: str = Field(..., min_length=1, description="ID cuộc hội thoại.")
    new_question: str = Field(
        ...,
        min_length=1,
        max_length=settings.MAX_QUESTION_LENGTH,
        description="Câu hỏi mới.",
    )
    history: List[ChatMessage] = Field(
        default_factory=list,
        max_length=settings.MAX_HISTORY_MESSAGES,
        description="Lịch sử trước đó. Caller nên tự cắt bớt trước khi gửi.",
    )
    stream: bool = False
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    max_new_tokens: Optional[int] = Field(default=None, ge=1, le=8192)

    # Bỏ qua các field lạ thay vì báo lỗi, giúp client cũ tương thích.
    model_config = ConfigDict(extra="ignore")


class BatchChatRequest(BaseModel):
    """
    Đầu vào cho batch inference: nhiều ``ChatRequest`` trong một lần gọi.

    Attributes:
        items: Danh sách các lượt hỏi - đáp cần xử lý.
        stream: Nếu True, trả về một SSE stream gộp cho tất cả item; mỗi sự
            kiện mang thêm ``index`` để client tách luồng.
    """

    items: List[ChatRequest] = Field(..., min_length=1)
    stream: bool = False

    model_config = ConfigDict(extra="ignore")

    @field_validator("items")
    @classmethod
    def _check_batch_size(cls, value: List[ChatRequest]) -> List[ChatRequest]:
        if len(value) > settings.MAX_BATCH_SIZE:
            raise ValueError(
                f"Số item vượt giới hạn MAX_BATCH_SIZE={settings.MAX_BATCH_SIZE}."
            )
        return value


class ChatResponse(BaseModel):
    """Kết quả của một lượt hỏi - đáp khi không streaming."""

    chat_id: str
    answer: str
    messages: List[ChatMessage] = Field(
        default_factory=list,
        description="History đã cập nhật (gồm cả câu hỏi gốc và câu trả lời).",
    )
    contexts: List[str] = Field(
        default_factory=list, description="Các chunk RAG đã dùng (để minh bạch/debug)."
    )


class BatchChatResponse(BaseModel):
    """Kết quả của một batch khi không streaming."""

    results: List[ChatResponse] = Field(default_factory=list)


class StreamEvent(BaseModel):
    """
    Một sự kiện trong SSE stream.

    ``type`` nhận một trong các giá trị:
    - ``start``: bắt đầu một item (kèm ``chat_id``, ``index``).
    - ``delta``: một đoạn text mới sinh ra (``delta``).
    - ``end``: kết thúc một item (``messages`` là history đã cập nhật,
      ``contexts`` là các đoạn RAG đã dùng).
    - ``error``: lỗi khi xử lý item (``error`` chứa thông báo).
    """

    type: Literal["start", "delta", "end", "error"]
    chat_id: Optional[str] = None
    index: Optional[int] = Field(default=None, description="Vị trí item trong batch.")
    delta: Optional[str] = None
    error: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    # Đoạn RAG đã truy hồi, cùng định dạng với ``ChatResponse.contexts``.
    # Có mặt ở event ``end`` để client hiển thị "nguồn tham khảo" ngay cả khi
    # đang stream — trước đó client bắt buộc phải gọi lại không-stream để có nó.
    contexts: Optional[List[str]] = None
