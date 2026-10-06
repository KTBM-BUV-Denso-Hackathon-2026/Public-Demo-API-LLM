"""
Cấu hình tập trung cho microservice ``llm_api``.

Mọi tham số đều có thể override bằng biến môi trường (hoặc file ``.env``
đặt cạnh ``main.py``). Xem ``.env.example`` để biết danh sách đầy đủ.

Quy ước: các giá trị mặc định ở đây phản ánh cấu hình đang dùng trong repo
(LMDeploy server trên cổng 23333, model Qwen2.5-7B-Instruct).
"""

import os

import dotenv

dotenv.load_dotenv()


def _get_str(name: str, default: str) -> str:
    """Đọc biến môi trường dạng chuỗi, trả về ``default`` nếu chưa set."""
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _get_int(name: str, default: int) -> int:
    """Đọc biến môi trường dạng số nguyên."""
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _get_float(name: str, default: float) -> float:
    """Đọc biến môi trường dạng số thực."""
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def _get_bool(name: str, default: bool) -> bool:
    """Đọc biến môi trường dạng boolean (true/1/yes/on)."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Kết nối tới LMDeploy Server (OpenAI-compatible API)
# ---------------------------------------------------------------------------
LM_DEPLOY_BASE_URL: str = _get_str("LM_DEPLOY_BASE_URL", "http://lmdeploy-server:23333/v1")
LM_DEPLOY_API_KEY: str = _get_str("LM_DEPLOY_API_KEY", "EMPTY")
LLM_MODEL: str = _get_str("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# Tham số sinh mặc định (có thể override theo từng request).
DEFAULT_TEMPERATURE: float = _get_float("DEFAULT_TEMPERATURE", 0.7)
DEFAULT_TOP_P: float = _get_float("DEFAULT_TOP_P", 0.8)
DEFAULT_MAX_NEW_TOKENS: int = _get_int("DEFAULT_MAX_NEW_TOKENS", 512)
LLM_REQUEST_TIMEOUT: float = _get_float("LLM_REQUEST_TIMEOUT", 120.0)

# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------
# Bật/tắt truy hồi RAG. Tắt để chạy API như một LLM proxy thuần.
RAG_ENABLED: bool = _get_bool("RAG_ENABLED", True)
RAG_TOP_K: int = _get_int("RAG_TOP_K", 5)

# Bảng chứa chunk + vector trong vector DB. Khớp với ``db_table`` của model
# ``RagChunkData`` trong ``denso_backend/apps/rag/models.py``.
RAG_CHUNK_TABLE: str = _get_str("RAG_CHUNK_TABLE", "rag_chunkdata")

# Model embedding dùng để vector hoá câu hỏi. PHẢI khớp với model đã dùng
# khi index dữ liệu, nếu không kết quả tìm kiếm sẽ sai.
EMBEDDING_MODEL: str = _get_str(
    "EMBEDDING_MODEL", "bkai-foundation-models/vietnamese-bi-encoder"
)
# Số chiều vector. 0 = tự động lấy theo model (không kiểm tra).
EMBEDDING_DIM: int = _get_int("EMBEDDING_DIM", 0)
# Nếu True: raise lỗi khi số chiều embedding khác ``EMBEDDING_DIM``.
# Nên bật ở production để tránh âm thầm tìm kiếm sai chiều.
STRICT_EMBEDDING_DIM: bool = _get_bool("STRICT_EMBEDDING_DIM", False)
EMBEDDING_BATCH_SIZE: int = _get_int("EMBEDDING_BATCH_SIZE", 32)

# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------
# Số item tối đa trong một request batch.
MAX_BATCH_SIZE: int = _get_int("MAX_BATCH_SIZE", 32)

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
# Chat history được lưu ở phía service gọi (denso_backend - model ``Chat``).
# ``llm_api`` nhận ``chat_id`` để định danh cuộc hội thoại và trả lại đầy đủ
# ``messages`` sau khi sinh xong để caller tự persist. Xem README để biết
# hợp đồng tích hợp (integration contract).
