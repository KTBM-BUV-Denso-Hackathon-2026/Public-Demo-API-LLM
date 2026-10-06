# llm_api — RAG + LLM Microservice

Microservice FastAPI cung cấp API hỏi đáp có RAG, hỗ trợ **batch inference**,
**RAG search theo batch** và **streaming (SSE)** tới LMDeploy Server.

## 1. Kiến trúc

```
Client (denso_backend / frontend)
        │  POST /chat | /chat/batch   { chat_id, new_question, history, stream }
        ▼
┌──────────────────────────────────────────────┐
│ llm_api (FastAPI)                            │
│                                              │
│  main.py      routes + SSE                   │
│  services.py  orchestration (RAG → prompt →  │
│               LLM, single/batch/stream)      │
│  schemas.py   Pydantic request/response      │
│  settings.py  cấu hình từ env                │
│                                              │
│  embedder/engine.py      cache + batch encode│
│  database/controllers.py vector search       │
│  llm/client.py           AsyncOpenAI wrapper │
└──────────────────────────────────────────────┘
        │                        │
        ▼                        ▼
  Vector DB (pgvector)     LMDeploy Server
  rag_chunkdata            (continuous batching)
```

Luồng xử lý một request:

1. **RAG search**: vector hoá `new_question` (batch encode nếu là batch) và
   truy hồi `top_k` chunk gần nhất từ `rag_chunkdata` theo cosine distance.
2. **Dựng prompt**: chèn context vào user message mới nhất, giữ nguyên history.
3. **Gọi LLM**: gửi tới LMDeploy ở chế độ non-stream hoặc stream.
4. **Trả kết quả**: JSON (kèm `messages` đã cập nhật) hoặc SSE.

## 2. Hợp đồng dữ liệu (integration contract)

`llm_api` **stateless** đối với chat history: nó không lưu DB hội thoại.
Caller (denso_backend) chịu trách nhiệm lưu/đọc history.

| Khái niệm | Ý nghĩa |
| --- | --- |
| `chat_id` | ID cuộc hội thoại (tương ứng `Chat.id` của Django). Thay cho `user_id` trước đây. Dùng để định danh/trace và làm khoá nếu sau này cần persist. |
| `history` | Danh sách message trước đó do caller truyền vào (`[{role, content}]`). |
| `messages` (response) | History sau khi thêm câu hỏi gốc + câu trả lời; caller lưu lại (ví dụ vào `Chat.message`). |

> Lưu ý: `llm_api` **không** đọc trực tiếp bảng `Chat` của Django vì field
> `Chat.message` là `EncryptedJSONField` — đọc thô từ service khác sẽ sai.

## 3. API

### `GET /health`

Trả `{"status": "ok"}`.

### `POST /chat`

Body (`ChatRequest`):

```json
{
  "chat_id": "12",
  "new_question": "Quy trình bảo trì định kỳ gồm những bước nào?",
  "history": [
    {"role": "user", "content": "Xin chào"},
    {"role": "assistant", "content": "Chào bạn, tôi có thể giúp gì?"}
  ],
  "stream": false,
  "top_k": 5,
  "temperature": 0.7
}
```

- `stream=false` → `ChatResponse`:

```json
{
  "chat_id": "12",
  "answer": "Quy trình gồm 3 bước ...",
  "messages": [
    {"role": "user", "content": "Xin chào"},
    {"role": "assistant", "content": "Chào bạn, tôi có thể giúp gì?"},
    {"role": "user", "content": "Quy trình bảo trì định kỳ gồm những bước nào?"},
    {"role": "assistant", "content": "Quy trình gồm 3 bước ..."}
  ],
  "contexts": ["- chunk 1 ...\n- chunk 2 ..."]
}
```

- `stream=true` → `text/event-stream`, mỗi event:

```
data: {"type": "start", "chat_id": "12"}

data: {"type": "delta", "chat_id": "12", "delta": "Quy"}

data: {"type": "delta", "chat_id": "12", "delta": " trình"}

data: {"type": "end", "chat_id": "12", "messages": [ ... ]}
```

### `POST /chat/batch`

Body (`BatchChatRequest`) — nhiều item trong một lần gọi:

```json
{
  "stream": false,
  "items": [
    {"chat_id": "12", "new_question": "Câu hỏi A", "history": []},
    {"chat_id": "13", "new_question": "Câu hỏi B", "history": []}
  ]
}
```

- `stream=false` → `BatchChatResponse` (`{"results": [ChatResponse, ...]}`),
  giữ đúng thứ tự đầu vào. RAG được truy hồi **một lần cho cả batch**.
- `stream=true` → **một** SSE stream gộp; mỗi event có thêm `index` để client
  tách luồng:

```
data: {"type": "start", "index": 0, "chat_id": "12"}
data: {"type": "start", "index": 1, "chat_id": "13"}
data: {"type": "delta", "index": 0, "chat_id": "12", "delta": "..."}
data: {"type": "delta", "index": 1, "chat_id": "13", "delta": "..."}
data: {"type": "end", "index": 0, "chat_id": "12", "messages": [ ... ]}
data: {"type": "end", "index": 1, "chat_id": "13", "messages": [ ... ]}
```

### Mã lỗi

| Mã | Ý nghĩa |
| --- | --- |
| `422` | Body sai (thiếu field, batch vượt `MAX_BATCH_SIZE`, `top_k` ngoài khoảng). |
| `502` | Lỗi từ RAG (vector DB/embedding) hoặc từ LMDeploy. |

## 4. Cấu hình

Toàn bộ tham số đọc từ biến môi trường / `.env`. Xem [.env.example](.env.example).
Các biến thường dùng: `LM_DEPLOY_BASE_URL`, `LLM_MODEL`, `RAG_ENABLED`,
`RAG_TOP_K`, `RAG_CHUNK_TABLE`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`,
`STRICT_EMBEDDING_DIM`, `MAX_BATCH_SIZE`, `DATABASE_URL`.

## 5. Chạy & test

```bash
pip install -r requirements.txt
cp .env.example .env        # rồi chỉnh các giá trị cho đúng môi trường
uvicorn main:app --host 0.0.0.0 --port 8000
```

Test (không cần DB và LLM thật):

```bash
python -m unittest discover -s tests -v
```

## 6. Giả định & giới hạn đã biết

1. **Số chiều embedding phải khớp** giữa model truy vấn và cột `vector` trong
   DB. Model mặc định `vietnamese-bi-encoder` trả 768 chiều, trong khi
   `RagChunkData.vector` (denso_backend) khai báo 1536 chiều → cần thống nhất
   trước khi chạy thật. Bật `STRICT_EMBEDDING_DIM=true` để phát hiện sớm.
2. **Truy hồi hiện dùng vector search thuần** trên `rag_chunkdata`. Cột `fts`
   (full-text) đã có sẵn cho hybrid search nhưng chưa bật.
3. `llm_api` không persist chat history; caller phải lưu `messages` trả về.
4. `handle_multiple_requests` cũ (hàm rời, không phải endpoint) đã được thay
   bằng endpoint `/chat/batch` + `services.answer_batch`.
