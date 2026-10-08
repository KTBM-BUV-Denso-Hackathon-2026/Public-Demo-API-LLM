"""Package chứa test cho microservice llm_api.

Test chạy hoàn toàn offline: không cần vector DB, LMDeploy, và cũng không cần
biết ``INTERNAL_API_TOKEN`` thật. Vì vậy ta tắt auth nội bộ **mặc định** cho cả
bộ test; ``tests/test_api.py::SecurityTests`` tự bật lại bằng ``mock.patch``
để kiểm tra đúng hành vi bật/tắt.

Làm vậy giúp test không phụ thuộc vào file ``.env`` của máy đang chạy.
"""

import settings

# Rỗng = auth tắt -> endpoint POST không đòi header X-Internal-Token.
settings.INTERNAL_API_TOKEN = ""
