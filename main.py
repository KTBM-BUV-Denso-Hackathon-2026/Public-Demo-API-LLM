import asyncio
from fastapi import FastAPI
from openai import AsyncOpenAI
from fastapi.responses import StreamingResponse
import psycopg2
from database.controllers import batch_search_rag_data, save_to_raw_data, save_to_rag_data, save_to_embedding_data, search_rag_data

# Client kết nối tới LMDeploy Server
lmdeploy_client = AsyncOpenAI(
    api_key="EMPTY",
    base_url="http://lmdeploy-server:23333/v1" # IP/Domain của con GPU LMDeploy
)

app = FastAPI()

@app.post("/chat")
async def handle_user_request(user_id: str, history: list, new_question: str):
    """
    Service này nhận request từ client
    """
    # 2. RAG Search: Query pgvector để lấy Context
    context = await search_rag_data(new_question)

    # 3. Tạo payload tin nhắn gửi cho LMDeploy:
    # Bản tin nhắn gửi cho LMDeploy thì KÈM RAG Context
    rag_user_message = {
        "role": "user",
        "content": f"THÔNG TIN THAM KHẢO:\n{context}\n\nCÂU HỎI: {new_question}"
    }
    messages_for_llm = history + [rag_user_message]

    # 4. Gửi tới LMDeploy Server (LMDeploy tự động Continuous Batching trên GPU)
    response = await lmdeploy_client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=messages_for_llm,
        temperature=0.7
    )
    ai_answer = response.choices[0].message.content

    # 5. LƯU VÀO DB: Chỉ lưu câu hỏi GỐC + câu trả lời của AI
    clean_user_message = {"role": "user", "content": new_question}
    clean_ai_message = {"role": "assistant", "content": ai_answer}
    
    await save_chat_history_to_db(user_id, [clean_user_message, clean_ai_message])

    return ai_answer

# Khi có N người dùng gửi request cùng lúc đến Service:
async def handle_multiple_requests(requests_list):
    # Tất cả các request này được gửi đồng thời sang LMDeploy Server
    # Engine của LMDeploy sẽ gom (batch) chúng lại trên GPU xử lý 1 lượt cực nhanh
    tasks = [handle_user_request(req["user_id"], req["question"]) for req in requests_list]
    results = await asyncio.gather(*tasks)
    return results