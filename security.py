"""Xác thực nội bộ giữa các microservice.

``llm_api`` và ``rag_worker`` KHÔNG có user/JWT của riêng mình — chúng chỉ được
gọi bởi ``denso_backend``. Nếu không có lớp bảo vệ này thì bất kỳ ai biết cổng
đều có thể:

- gọi ``/chat`` để đốt tài nguyên LLM,
- gọi ``/ingest`` để **xoá và ghi lại** chunk của bất kỳ tài liệu nào.

Cơ chế: header ``X-Internal-Token`` phải khớp ``INTERNAL_API_TOKEN``.

Hành vi khi ``INTERNAL_API_TOKEN`` **để trống**: auth bị TẮT (kèm cảnh báo rõ
ràng lúc khởi động). Điều này giữ được trải nghiệm "clone rồi chạy ngay" cho dev,
nhưng trong ``.env`` thật và trên production thì biến này luôn được đặt.
"""

from __future__ import annotations

import hmac
import logging
from typing import Optional

from fastapi import Header, HTTPException, status

import settings

logger = logging.getLogger(__name__)

#: Tên header mà ``denso_backend`` gửi kèm mọi request nội bộ.
HEADER_NAME = "X-Internal-Token"

#: 401 khi thiếu/sai token. Client nội bộ gặp lỗi này nghĩa là cấu hình lệch.
_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail=f"Thiếu hoặc sai header {HEADER_NAME} (kiểm tra INTERNAL_API_TOKEN).",
)

_warned_disabled = False


def is_enabled() -> bool:
    """True khi đã cấu hình ``INTERNAL_API_TOKEN`` (tức auth đang bật)."""
    return bool(settings.INTERNAL_API_TOKEN)


def warn_if_disabled(service: str) -> None:
    """Cảnh báo một lần nếu đang chạy mà chưa bật auth."""
    global _warned_disabled
    if not is_enabled() and not _warned_disabled:
        _warned_disabled = True
        logger.warning(
            "⚠ %s đang chạy KHÔNG có xác thực nội bộ (INTERNAL_API_TOKEN rỗng). "
            "Chỉ nên dùng ở localhost; production bắt buộc đặt INTERNAL_API_TOKEN.",
            service,
        )


async def require_internal_token(
    x_internal_token: Optional[str] = Header(default=None, alias=HEADER_NAME),
) -> None:
    """FastAPI dependency: chặn request không hợp lệ.

    Dùng ``Optional[str]`` chứ không phải ``str | None`` vì FastAPI phải *đánh giá
    thật* annotation này để đọc tên header, mà toán tử ``|`` cho union chỉ có từ
    Python 3.10 (workspace đang dùng 3.9).

    Raises:
        HTTPException 401: khi auth đang bật mà header thiếu hoặc sai.
    """
    expected = settings.INTERNAL_API_TOKEN
    if not expected:
        # Auth tắt -> cho qua (chế độ dev).
        return
    if not x_internal_token or not hmac.compare_digest(x_internal_token, expected):
        raise _UNAUTHORIZED
