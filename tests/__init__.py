"""Package chứa test cho microservice llm_api.

Bộ test phải **chạy hoàn toàn offline và xác định**: không cần vector DB,
LMDeploy, Ollama hay bất kỳ server LLM nào đang chạy.

Vì vậy, khi import package này ta ép hai thứ:

- ``INTERNAL_API_TOKEN = ""``  -> endpoint POST không đòi header token.
- ``LLM_PROVIDER = "mock"``    -> provider giả lập, không gọi ra ngoài.

Đây là LƯỚI AN TOÀN: nếu một test quên mock ``services._complete`` /
``services._stream`` thì nó vẫn chạy nhanh và ổn định, thay vì âm thầm gọi
LLM thật (làm test chậm hàng giây và phụ thuộc máy đang chạy).
Các test cố ý kiểm tra hành vi provider thật sẽ tự `mock.patch` lại biến này.
"""

import settings

# Rỗng = auth tắt -> endpoint POST không đòi header X-Internal-Token.
settings.INTERNAL_API_TOKEN = ""

# Ép provider giả lập để test không bao giờ gọi LLM thật (xem docstring).
settings.LLM_PROVIDER = "mock"
