"""
Kết nối tới PostgreSQL + pgvector (CHUNG với ``denso_backend``).

Hệ thống dùng **một** database duy nhất cho cả bảng nghiệp vụ lẫn bảng vector.
Lý do: ``rag_rawdata.user_id`` là khoá ngoại trỏ tới ``core_users`` mà PostgreSQL
không hỗ trợ khoá ngoại xuyên database, nên tách "vector DB" riêng sẽ khiến
migration của app ``rag`` không chạy được. Vì vậy chỉ cần đúng một biến
``DATABASE_URL`` — phải trỏ vào **cùng** DB mà ``denso_backend`` dùng.

Dùng ``ThreadedConnectionPool`` để tái sử dụng kết nối giữa các request thay vì
mở/đóng liên tục — quan trọng vì mỗi lần connect tới Postgres cloud mất hàng
trăm ms.

``get_db()`` là context manager, dùng như sau::

    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(...)
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

import psycopg2
from psycopg2 import pool as psycopg2_pool

import settings

logger = logging.getLogger(__name__)

_pool: Optional[psycopg2_pool.ThreadedConnectionPool] = None
_pool_lock = threading.Lock()


def _database_url() -> str:
    """Trả về URL kết nối DB, báo lỗi rõ nếu chưa cấu hình."""
    url = settings.DATABASE_URL
    if not url:
        raise RuntimeError(
            "Chưa cấu hình DATABASE_URL cho llm_api. "
            "Phải trỏ TỚI CÙNG database với denso_backend. "
            "Xem llm_api/.env.example."
        )
    return url


def get_pool() -> psycopg2_pool.ThreadedConnectionPool:
    """Trả về connection pool dùng chung (khởi tạo lần đầu khi cần)."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = psycopg2_pool.ThreadedConnectionPool(
                    minconn=settings.DB_POOL_MIN,
                    maxconn=settings.DB_POOL_MAX,
                    dsn=_database_url(),
                )
                logger.info(
                    "Đã tạo connection pool tới DB (min=%d, max=%d)",
                    settings.DB_POOL_MIN,
                    settings.DB_POOL_MAX,
                )
    return _pool


def close_pool() -> None:
    """Đóng toàn bộ kết nối trong pool (dùng khi shutdown/test)."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.closeall()
            _pool = None


def _borrow_live_connection(pool, attempts: int = 2):
    """Mượn một kết nối **còn sống** từ pool, loại bỏ kết nối đã chết.

    Vì sao cần?
    ``DATABASE_URL`` đang trỏ tới connection **POOLER** của cloud Postgres
    (Neon/Supabase — hostname có chữ ``-pooler``). Pooler chủ động đóng kết nối
    rảnh sau một thời gian, nhưng kết nối đó vẫn nằm trong pool của ta. Lần
    mượn kế tiếp sẽ dùng lại đúng socket đã chết và mọi query sẽ nổ:

        psycopg2.OperationalError: SSL SYSCALL error:
        Software caused connection abort (0x00002745/10053)

    psycopg2 (khác với ``psycopg_pool`` của psycopg3, và khác SQLAlchemy)
    **không hỗ trợ** ``pool_pre_ping``. Vì vậy ta tự ping rẻ bằng ``SELECT 1``
    trước khi dùng: hỏng thì loại bỏ, thử lại tối đa ``attempts`` lần.
    """
    last_error = None
    for _ in range(max(1, attempts)):
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return conn
        except Exception as exc:  # noqa: BLE001 - cần bắt mọi kiểu lỗi kết nối
            last_error = exc
            logger.warning(
                "Kết nối DB lấy từ pool không dùng được, sẽ bỏ và thử lại: %s", exc
            )
            try:
                pool.putconn(conn, close=True)
            except Exception:  # pragma: no cover - pool đang hỏng
                pass
    if last_error is not None:
        raise last_error
    raise RuntimeError("Không lấy được kết nối DB nào khỏe.")


@contextmanager
def get_db() -> Iterator["psycopg2.extensions.connection"]:
    """Mượn một kết nối từ pool, tự trả lại sau khi dùng xong.

    - Kiểm tra kết nối còn sống trước khi dùng (xem ``_borrow_live_connection``).
    - Nếu trong block có exception → rollback trước khi trả kết nối về pool.
    - Kết nối hỏng/đã đóng sẽ bị loại khỏi pool thay vì tái sử dụng.
    """
    pool = get_pool()
    conn = _borrow_live_connection(
        pool, attempts=getattr(settings, "DB_PREPING_ATTEMPTS", 2)
    )
    broken = False
    try:
        yield conn
    except Exception:
        broken = True
        try:
            conn.rollback()
        except Exception:  # pragma: no cover - kết nối đã chết
            pass
        raise
    finally:
        try:
            if broken or conn.closed:
                pool.putconn(conn, close=True)
            else:
                # Dọn transaction còn sót để kết nối sạch khi trả về pool.
                if conn.get_transaction_status() != psycopg2.extensions.TRANSACTION_STATUS_IDLE:
                    conn.rollback()
                pool.putconn(conn)
        except Exception:  # pragma: no cover - không được che lỗi gốc
            logger.exception("Lỗi khi trả kết nối về pool")
