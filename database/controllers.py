"""
Truy hồi RAG trên PostgreSQL + pgvector (DB chung với ``denso_backend``).

Ghi chú kỹ thuật quan trọng:
- Vector embedding của chunk nằm ở bảng ``rag_chunkdata`` (cột ``vector``,
  kiểu ``vector(N)`` của pgvector).
- Toán tử ``<=>`` là cosine distance của pgvector; sắp xếp tăng dần cho ra
  các chunk gần nghĩa nhất.
- ``RAG_CHUNK_TABLE`` được nội suy từ thẻ truy hồi nên phải là hằng số an toàn
  (nó được nội suy thẳng vào SQL, không dùng placeholder của psycopg2).

Việc GHI dữ liệu vào bảng này do ``rag_worker`` đảm nhiệm, không nằm ở đây.
"""

from __future__ import annotations

import logging
import re
from typing import List, Sequence

import settings
from database.config import get_db

logger = logging.getLogger(__name__)

#: Tên bảng hợp lệ: chữ/số/gạch dưới, có thể có schema (vd ``public.rag_chunkdata``).
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


def _safe_table(table: str) -> str:
    """Xác thực tên bảng trước khi nội suy vào SQL.

    ``RAG_CHUNK_TABLE`` không thể dùng placeholder của psycopg2 (placeholder chỉ
    thay giá trị, không thay định danh), nên nó được ghép thẳng vào câu SQL. Vì
    vậy phải kiểm tra bằng allow-list để một giá trị cấu hình sai không biến thành
    SQL injection.

    Raises:
        ValueError: nếu tên bảng không hợp lệ.
    """
    if not _IDENTIFIER_RE.match(table or ""):
        raise ValueError(
            f"RAG_CHUNK_TABLE không hợp lệ: {table!r}. "
            "Chỉ chấp nhận dạng 'ten_bang' hoặc 'schema.ten_bang'."
        )
    return table


def _to_vector_literal(vector: Sequence[float]) -> str:
    """
    Chuyển một vector Python thành literal pgvector, ví dụ ``[0.1,-0.2,...]``.

    Dùng ``repr`` để giữ nguyên độ chính xác của số thực.
    """
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


def search_by_embeddings(
    embeddings: Sequence[Sequence[float]],
    top_k: int = 5,
) -> List[str]:
    """
    Truy hồi top-k chunk cho từng embedding đã tính trước.

    Tách riêng khỏi bước encode để có thể vector hoá cả batch trong một lần
    (xem ``batch_search_rag_data``) rồi tái sử dụng kết quả.

    Args:
        embeddings: Danh sách vector câu hỏi.
        top_k: Số chunk gần nhất lấy cho mỗi câu hỏi.

    Returns:
        Danh sách context string, mỗi phần tử ứng với một embedding đầu vào.
        Nếu không có chunk nào phù hợp, phần tử tương ứng là chuỗi rỗng.

    Raises:
        ValueError: nếu ``top_k`` không dương.
    """
    if top_k <= 0:
        raise ValueError("top_k phải là số nguyên dương.")

    table = _safe_table(settings.RAG_CHUNK_TABLE)
    query = (
        f"SELECT chunk_content FROM {table} "
        f"WHERE vector IS NOT NULL "
        f"ORDER BY vector <=> %s::vector "
        f"LIMIT %s"
    )

    contexts: List[str] = []
    with get_db() as conn:
        with conn.cursor() as cursor:
            for embedding in embeddings:
                cursor.execute(query, (_to_vector_literal(embedding), top_k))
                rows = cursor.fetchall()
                contexts.append("\n".join(f"- {row[0]}" for row in rows))
    return contexts


def search_rag_data(latest_query: str, top_k: int = 5) -> str:
    """
    Truy hồi context cho MỘT câu hỏi (encode + search).

    Args:
        latest_query: Câu hỏi cần tìm context.
        top_k: Số chunk gần nhất.

    Returns:
        Context string đã ghép các chunk, hoặc chuỗi rỗng nếu không có.
    """
    # Import muộn để tránh vòng lặp import (embedder cần settings, không cần DB).
    from embedder.engine import encode_texts

    embeddings = encode_texts([latest_query])
    contexts = search_by_embeddings(embeddings, top_k=top_k)
    return contexts[0] if contexts else ""


def batch_search_rag_data(latest_queries: Sequence[str], top_k: int = 5) -> List[str]:
    """
    Truy hồi context cho NHIỀU câu hỏi trong một lần encode (batch retrieval).

    Args:
        latest_queries: Danh sách câu hỏi.
        top_k: Số chunk gần nhất cho mỗi câu hỏi.

    Returns:
        Danh sách context string, giữ đúng thứ tự với ``latest_queries``.
    """
    from embedder.engine import encode_texts

    queries = list(latest_queries)
    if not queries:
        return []
    embeddings = encode_texts(queries)
    return search_by_embeddings(embeddings, top_k=top_k)
