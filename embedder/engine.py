"""
Engine embedding cho RAG.

Bọc ``sentence_transformers.SentenceTransformer`` với cache theo tên model
để tránh nạp lại model (rất tốn thời gian/bộ nhớ) ở mỗi request.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from sentence_transformers import SentenceTransformer

import settings

logger = logging.getLogger(__name__)

# Cache model theo tên: {model_name: SentenceTransformer}
_MODELS: Dict[str, SentenceTransformer] = {}
# Khi nhiều luồng cùng nạp model lần đầu, dùng lock để tránh nạp trùng.
_LOCK = threading.Lock()


def get_model(model_name: Optional[str] = None) -> SentenceTransformer:
    """
    Trả về ``SentenceTransformer`` đã cache cho ``model_name``.

    Model chỉ được nạp một lần cho mỗi tên; các lần gọi sau lấy từ cache.
    An toàn với đa luồng (double-checked locking).

    Args:
        model_name: Tên model; mặc định ``settings.EMBEDDING_MODEL``.

    Returns:
        Đối tượng ``SentenceTransformer`` đã sẵn sàng dùng.
    """
    name = model_name or settings.EMBEDDING_MODEL
    model = _MODELS.get(name)
    if model is not None:
        return model

    with _LOCK:
        # Kiểm tra lại sau khi lấy lock (double-checked locking).
        model = _MODELS.get(name)
        if model is None:
            logger.info("Đang nạp embedding model: %s", name)
            model = SentenceTransformer(name)
            _MODELS[name] = model
    return model


def get_model_dimension(model_name: Optional[str] = None) -> Optional[int]:
    """Trả về số chiều vector của model, hoặc ``None`` nếu model không hỗ trợ."""
    model = get_model(model_name)
    try:
        return int(model.get_sentence_embedding_dimension())
    except Exception:  # pragma: no cover - phụ thuộc phiên bản thư viện
        return None


def encode_texts(
    texts: List[str],
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    normalize: bool = True,
) -> List[List[float]]:
    """
    Vector hoá một danh sách văn bản trong MỘT lần gọi (batch encode).

    Hàm này chạy đồng bộ và tốn CPU/GPU; khi gọi trong async context nên
    bọc bằng ``asyncio.to_thread`` để không chặn event loop.

    Args:
        texts: Danh sách văn bản cần vector hoá.
        model_name: Tên model embedding; mặc định theo settings.
        batch_size: Kích thước batch encode; mặc định theo settings.
        normalize: Chuẩn hoá L2 để dùng cosine distance ổn định.

    Returns:
        Ma trận embedding dạng list[list[float]] (mỗi phần tử là 1 vector).

    Raises:
        RuntimeError: Nếu số chiều embedding không khớp ``settings.EMBEDDING_DIM``
            khi ``settings.STRICT_EMBEDDING_DIM`` được bật.
    """
    if not texts:
        return []

    model = get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size or settings.EMBEDDING_BATCH_SIZE,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    vectors = [vector.tolist() for vector in embeddings]

    _validate_dimension(len(vectors[0]) if vectors else 0)
    return vectors


def _validate_dimension(dimension: int) -> None:
    """
    Kiểm tra số chiều embedding so với cấu hình.

    Ghi cảnh báo khi lệch; raise lỗi nếu ``STRICT_EMBEDDING_DIM`` được bật.
    Việc này chặn tình huống âm thầm truy vấn sai chiều với vector DB.
    """
    expected = settings.EMBEDDING_DIM
    if expected <= 0 or dimension == expected:
        return

    message = (
        f"Embedding dimension mismatch: model trả về {dimension} chiều "
        f"nhưng EMBEDDING_DIM={expected}. Kiểm tra lại model embedding và "
        f"cột vector trong vector DB."
    )
    if settings.STRICT_EMBEDDING_DIM:
        raise RuntimeError(message)
    logger.warning(message)


def reset_cache() -> None:
    """Xoá cache model (chủ yếu dùng cho test)."""
    with _LOCK:
        _MODELS.clear()
