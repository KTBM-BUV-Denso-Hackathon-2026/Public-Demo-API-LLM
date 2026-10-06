"""
Truy vấn RAG trên vector DB (PostgreSQL + pgvector).

File này gồm hai nhóm hàm:

1. Nhóm ghi dữ liệu (``save_to_*``): dùng cho pipeline nạp dữ liệu.
2. Nhóm truy hồi (``search_*``, ``search_by_embeddings``): dùng cho lúc
   inference để lấy context đưa vào prompt.

Ghi chú kỹ thuật quan trọng:
- Vector embedding của chunk nằm ở bảng ``rag_chunkdata`` (cột ``vector``,
  kiểu ``vector(N)`` của pgvector), KHÔNG nằm ở ``rag_ragdata``.
- Toán tử ``<=>`` là cosine distance của pgvector; sắp xếp tăng dần cho ra
  các chunk gần nghĩa nhất.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import settings
from database.config import get_db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Nhóm ghi dữ liệu (ingestion)
# ---------------------------------------------------------------------------
def save_to_raw_data(columns: Sequence[str], values: Sequence) -> None:
    """
    Ghi một bản ghi vào bảng ``rag_rawdata``.

    Args:
        columns: Danh sách tên cột.
        values: Danh sách giá trị tương ứng với ``columns``.
    """
    _insert("rag_rawdata", columns, values)


def save_to_rag_data(columns: Sequence[str], values: Sequence) -> None:
    """
    Ghi một bản ghi vào bảng ``rag_ragdata``.

    Args:
        columns: Danh sách tên cột.
        values: Danh sách giá trị tương ứng với ``columns``.
    """
    _insert("rag_ragdata", columns, values)


def save_to_embedding_data(columns: Sequence[str], values: Sequence) -> None:
    """
    Ghi một chunk (kèm vector) vào bảng ``rag_chunkdata``.

    Lưu ý: bảng ``rag_chunkdata`` có tập cột cố định, nên câu lệnh dùng
    danh sách cột cố định thay vì ``columns`` truyền vào.

    Args:
        columns: Danh sách tên cột (giữ để tương thích chữ ký hàm cũ).
        values: Giá trị theo thứ tự
            ``(id, chunk_content, chunk_index, metadata, vector, fts)``.
    """
    query = (
        "INSERT INTO rag_chunkdata "
        "(id, chunk_content, chunk_index, metadata, vector, fts) "
        "VALUES (%s, %s, %s, %s, %s, %s)"
    )
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, values)
            conn.commit()


def _insert(table: str, columns: Sequence[str], values: Sequence) -> None:
    """
    Hàm ghi dùng chung: tạo câu INSERT động an toàn bằng placeholder ``%s``.

    ``table`` và tên cột KHÔNG đến từ input người dùng nên an toàn với
    SQL injection; giá trị luôn truyền qua placeholder.
    """
    placeholders = ", ".join(["%s"] * len(values))
    columns_str = ", ".join(columns)
    query = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, values)
            conn.commit()


# ---------------------------------------------------------------------------
# Nhóm truy hồi (retrieval)
# ---------------------------------------------------------------------------
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
    """
    if top_k <= 0:
        raise ValueError("top_k phải là số nguyên dương.")

    table = settings.RAG_CHUNK_TABLE
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
    Truy hồi context cho MỘT câu hỏi.

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
