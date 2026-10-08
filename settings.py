"""
Cấu hình tập trung cho microservice ``llm_api``.

Mọi tham số đều override được bằng biến môi trường (hoặc file ``.env`` đặt
cạnh ``main.py``). Xem ``.env.example`` để biết danh sách đầy đủ.
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
# Nhà cung cấp LLM
# ---------------------------------------------------------------------------
# ``openai_compatible``: gọi LMDeploy/vLLM/OpenAI qua HTTP (mặc định, production).
# ``mock``             : trả lời giả lập, KHÔNG cần GPU — chỉ dùng cho dev/test.
LLM_PROVIDER: str = _get_str("LLM_PROVIDER", "openai_compatible").strip().lower()

LM_DEPLOY_BASE_URL: str = _get_str("LM_DEPLOY_BASE_URL", "http://localhost:23333/v1")
LM_DEPLOY_API_KEY: str = _get_str("LM_DEPLOY_API_KEY", "EMPTY")
LLM_MODEL: str = _get_str("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# Tham số sinh mặc định (có thể override theo từng request).
DEFAULT_TEMPERATURE: float = _get_float("DEFAULT_TEMPERATURE", 0.7)
DEFAULT_TOP_P: float = _get_float("DEFAULT_TOP_P", 0.8)
DEFAULT_MAX_NEW_TOKENS: int = _get_int("DEFAULT_MAX_NEW_TOKENS", 512)
LLM_REQUEST_TIMEOUT: float = _get_float("LLM_REQUEST_TIMEOUT", 120.0)

# Độ trễ giả lập cho từng token khi dùng provider ``mock`` (ms).
MOCK_STREAM_DELAY_MS: int = _get_int("MOCK_STREAM_DELAY_MS", 0)

# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------
# Bật/tắt truy hồi RAG. Tắt để chạy API như một LLM proxy thuần.
RAG_ENABLED: bool = _get_bool("RAG_ENABLED", True)
RAG_TOP_K: int = _get_int("RAG_TOP_K", 5)

# Bảng chứa chunk + vector trong DB (chung với denso_backend). Khớp ``db_table``
# của model ``RagChunkData`` trong ``denso_backend/apps/rag/models.py``.
# Tên bảng được nội suy thẳng vào SQL nên phải hợp lệ theo allow-list trong
# ``database/controllers.py::_safe_table``.
RAG_CHUNK_TABLE: str = _get_str("RAG_CHUNK_TABLE", "rag_chunkdata")

# Model embedding dùng để vector hoá câu hỏi. PHẢI khớp với model đã dùng khi
# index dữ liệu (``rag_worker``), nếu không kết quả tìm kiếm sẽ sai.
EMBEDDING_MODEL: str = _get_str(
    "EMBEDDING_MODEL", "bkai-foundation-models/vietnamese-bi-encoder"
)
# Số chiều vector. 0 = tự động lấy theo model (không kiểm tra).
EMBEDDING_DIM: int = _get_int("EMBEDDING_DIM", 768)
# Nếu True: raise lỗi khi số chiều embedding khác ``EMBEDDING_DIM``.
# Bật để tránh âm thầm truy vấn sai chiều với cột ``vector`` trong DB.
STRICT_EMBEDDING_DIM: bool = _get_bool("STRICT_EMBEDDING_DIM", True)
EMBEDDING_BATCH_SIZE: int = _get_int("EMBEDDING_BATCH_SIZE", 32)

# ---------------------------------------------------------------------------
# Database (PostgreSQL + pgvector) — CHUNG với denso_backend
# ---------------------------------------------------------------------------
DATABASE_URL: str = _get_str("DATABASE_URL", "")
DB_POOL_MIN: int = _get_int("DB_POOL_MIN", 1)
DB_POOL_MAX: int = _get_int("DB_POOL_MAX", 10)

# ---------------------------------------------------------------------------
# Xác thực nội bộ (xem security.py)
# ---------------------------------------------------------------------------
# Shared-secret mà denso_backend gửi kèm header ``X-Internal-Token``.
# Rỗng = tắt auth (chỉ dành cho localhost). Production BẮT BUỘC đặt.
INTERNAL_API_TOKEN: str = _get_str("INTERNAL_API_TOKEN", "")

# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------
# Số item tối đa trong một request batch.
MAX_BATCH_SIZE: int = _get_int("MAX_BATCH_SIZE", 32)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
# Rất quan trọng khi dùng model NHỎ (1.5B–3B): không có chỉ dẫn rõ, các model đó
# rất dễ trả lời lạch hoặc bịa số liệu kỹ thuật. System prompt dưới đây ép
# model chỉ dựa trên tài liệu được cung cấp và nói thẳng khi không có thông tin.
#
# Để trống -> không gửi system message (giữ hành vi cũ).
SYSTEM_PROMPT: str = _get_str("SYSTEM_PROMPT", "")

# ---------------------------------------------------------------------------
# Giới hạn kích thước đầu vào (chống prompt khổng lồ)
# ---------------------------------------------------------------------------
# Khớp với ``AskRequestSerializer.question`` của denso_backend (max_length=8000).
MAX_QUESTION_LENGTH: int = _get_int("MAX_QUESTION_LENGTH", 8000)
# Trần cứng cho số message lịch sử. denso_backend tự cắt bớt trước khi gửi
# (``LLM_MAX_HISTORY_MESSAGES``); con số này chỉ là lưới an toàn cuối.
MAX_HISTORY_MESSAGES: int = _get_int("MAX_HISTORY_MESSAGES", 200)
# Trần cứng cho độ dài một message trong lịch sử.
MAX_MESSAGE_LENGTH: int = _get_int("MAX_MESSAGE_LENGTH", 32000)

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
# ``llm_api`` KHÔNG lưu hội thoại: nó nhận ``chat_id`` để định danh và trả lại
# đầy đủ ``messages`` sau khi sinh xong để caller (denso_backend) tự persist.
# Xem README để biết hợp đồng tích hợp (integration contract).
