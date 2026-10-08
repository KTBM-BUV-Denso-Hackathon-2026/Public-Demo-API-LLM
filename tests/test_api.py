"""
Test cho ``llm_api`` chạy hoàn toàn offline.

Không cần vector DB và LMDeploy: các điểm tiêm ``services._complete`` /
``services._stream`` / ``services.retrieve_contexts*`` được mock.

Chạy từ thư mục ``llm_api``:

    python -m unittest discover -s tests -v
"""

import json
import unittest
from unittest import mock

from fastapi.testclient import TestClient

import services
import settings
from main import app


def _make_stream(tokens):
    """Tạo hàm streaming giả trả về lần lượt các token cho trước."""

    async def _fake_stream(messages, params):
        for token in tokens:
            yield token

    return _fake_stream


def _parse_sse(body):
    """Tách body SSE thành danh sách dict event."""
    events = []
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


class ApiTestCase(unittest.TestCase):
    """Base: tắt RAG để không đụng tới vector DB và khôi phục sau mỗi test."""

    def setUp(self):
        self.client = TestClient(app)
        self._rag_enabled = settings.RAG_ENABLED
        settings.RAG_ENABLED = False

    def tearDown(self):
        settings.RAG_ENABLED = self._rag_enabled


class TestHealth(ApiTestCase):
    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


class TestChatSingle(ApiTestCase):
    def test_non_stream_appends_history(self):
        """Non-stream: trả answer và history đã thêm 2 message."""
        fake_complete = mock.AsyncMock(return_value="Câu trả lời")
        payload = {
            "chat_id": "12",
            "new_question": "Câu hỏi mới",
            "history": [
                {"role": "user", "content": "Xin chào"},
                {"role": "assistant", "content": "Chào bạn"},
            ],
        }
        with mock.patch.object(services, "_complete", fake_complete):
            response = self.client.post("/chat", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["chat_id"], "12")
        self.assertEqual(data["answer"], "Câu trả lời")
        self.assertEqual(len(data["messages"]), 4)
        self.assertEqual(data["messages"][0]["content"], "Xin chào")
        # Chỉ lưu câu hỏi gốc, KHÔNG kèm context RAG.
        self.assertEqual(data["messages"][2]["content"], "Câu hỏi mới")
        self.assertEqual(data["messages"][3]["role"], "assistant")
        self.assertEqual(data["contexts"], [])

    def test_rag_context_is_injected_into_prompt(self):
        """Context RAG phải được chèn vào user message gửi cho LLM."""
        fake_complete = mock.AsyncMock(return_value="ANS")
        with mock.patch.object(
            services, "retrieve_contexts", mock.AsyncMock(return_value=["CTX-DATA"])
        ), mock.patch.object(services, "_complete", fake_complete):
            response = self.client.post(
                "/chat",
                json={"chat_id": "1", "new_question": "Quy trình?"},
            )

        self.assertEqual(response.status_code, 200)
        messages = fake_complete.call_args.args[0]
        self.assertTrue(messages[-1]["content"].startswith("THÔNG TIN THAM KHẢO:"))
        self.assertIn("CTX-DATA", messages[-1]["content"])
        self.assertTrue(messages[-1]["content"].endswith("CÂU HỎI: Quy trình?"))

    def test_stream_emits_start_delta_end(self):
        payload = {"chat_id": "7", "new_question": "Hi", "stream": True}
        with mock.patch.object(services, "_stream", _make_stream(["Xin ", "chào"])):
            response = self.client.post("/chat", json=payload)

        self.assertEqual(response.status_code, 200)
        events = _parse_sse(response.text)
        self.assertEqual(
            [event["type"] for event in events], ["start", "delta", "delta", "end"]
        )
        self.assertEqual("".join(e["delta"] for e in events if e["type"] == "delta"), "Xin chào")
        end_event = events[-1]
        self.assertEqual(len(end_event["messages"]), 2)
        self.assertEqual(end_event["messages"][1]["content"], "Xin chào")

    def test_stream_error_is_reported_as_event(self):
        """Lỗi khi sinh phải trả event type=error thay vì vỡ kết nối."""

        async def _boom(messages, params):
            raise RuntimeError("LMDeploy chết")
            yield  # pragma: no cover - để hàm là async generator

        with mock.patch.object(services, "_stream", _boom):
            response = self.client.post(
                "/chat", json={"chat_id": "1", "new_question": "Hi", "stream": True}
            )

        events = _parse_sse(response.text)
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("LMDeploy chết", events[-1]["error"])


class TestChatBatch(ApiTestCase):
    def test_non_stream_preserves_order_and_context(self):
        """Batch non-stream: giữ thứ tự, mỗi item nhận đúng context của nó."""
        fake_complete = mock.AsyncMock(
            side_effect=lambda messages, params: messages[-1]["content"]
        )
        items = [
            {"chat_id": "a", "new_question": "q1"},
            {"chat_id": "b", "new_question": "q2"},
        ]
        with mock.patch.object(
            services,
            "retrieve_contexts_varied",
            mock.AsyncMock(return_value=["CTX-A", "CTX-B"]),
        ), mock.patch.object(services, "_complete", fake_complete):
            response = self.client.post("/chat/batch", json={"items": items})

        self.assertEqual(response.status_code, 200)
        results = response.json()["results"]
        self.assertEqual([r["chat_id"] for r in results], ["a", "b"])
        self.assertIn("CTX-A", results[0]["answer"])
        self.assertTrue(results[0]["answer"].endswith("q1"))
        self.assertIn("CTX-B", results[1]["answer"])
        self.assertTrue(results[1]["answer"].endswith("q2"))

    def test_batch_size_limit_enforced(self):
        original = settings.MAX_BATCH_SIZE
        settings.MAX_BATCH_SIZE = 1
        try:
            response = self.client.post(
                "/chat/batch",
                json={
                    "items": [
                        {"chat_id": "a", "new_question": "q1"},
                        {"chat_id": "b", "new_question": "q2"},
                    ]
                },
            )
        finally:
            settings.MAX_BATCH_SIZE = original

        self.assertEqual(response.status_code, 422)

    def test_stream_events_are_indexed_per_item(self):
        items = [
            {"chat_id": "a", "new_question": "q1"},
            {"chat_id": "b", "new_question": "q2"},
            {"chat_id": "c", "new_question": "q3"},
        ]
        with mock.patch.object(services, "_stream", _make_stream(["tok"])):
            response = self.client.post(
                "/chat/batch", json={"stream": True, "items": items}
            )

        self.assertEqual(response.status_code, 200)
        events = _parse_sse(response.text)
        self.assertEqual(len(events), 9)  # 3 item × (start + delta + end)

        per_index = {}
        for event in events:
            per_index.setdefault(event["index"], []).append(event)
        self.assertEqual(set(per_index.keys()), {0, 1, 2})
        for index, item_events in per_index.items():
            self.assertEqual(
                [e["type"] for e in item_events], ["start", "delta", "end"]
            )
            self.assertEqual(item_events[0]["chat_id"], items[index]["chat_id"])


class TestValidation(ApiTestCase):
    def test_missing_question_rejected(self):
        response = self.client.post("/chat", json={"chat_id": "1"})
        self.assertEqual(response.status_code, 422)

    def test_blank_question_rejected(self):
        response = self.client.post(
            "/chat", json={"chat_id": "1", "new_question": ""}
        )
        self.assertEqual(response.status_code, 422)

    def test_top_k_out_of_range_rejected(self):
        response = self.client.post(
            "/chat", json={"chat_id": "1", "new_question": "q", "top_k": 0}
        )
        self.assertEqual(response.status_code, 422)

    def test_empty_batch_rejected(self):
        response = self.client.post("/chat/batch", json={"items": []})
        self.assertEqual(response.status_code, 422)

    def test_unknown_fields_ignored(self):
        """Field lạ không làm request lỗi (tương thích client cũ)."""
        with mock.patch.object(
            services, "_complete", mock.AsyncMock(return_value="ok")
        ):
            response = self.client.post(
                "/chat",
                json={
                    "chat_id": "1",
                    "new_question": "q",
                    "user_id": "legacy",
                    "foo": "bar",
                },
            )
        self.assertEqual(response.status_code, 200)


class TestPromptBuilding(unittest.TestCase):
    def test_empty_context_returns_question(self):
        self.assertEqual(services.build_rag_prompt("", "Câu hỏi"), "Câu hỏi")

    def test_context_is_wrapped(self):
        prompt = services.build_rag_prompt("- đoạn 1", "Câu hỏi")
        self.assertIn("THÔNG TIN THAM KHẢO:", prompt)
        self.assertIn("- đoạn 1", prompt)
        self.assertTrue(prompt.endswith("CÂU HỎI: Câu hỏi"))

    def test_resolve_params_uses_defaults(self):
        from schemas import ChatRequest

        request = ChatRequest(chat_id="1", new_question="q")
        params = services.resolve_params(request)
        self.assertEqual(params.temperature, settings.DEFAULT_TEMPERATURE)
        self.assertEqual(params.max_new_tokens, settings.DEFAULT_MAX_NEW_TOKENS)

    def test_resolve_params_respects_override(self):
        from schemas import ChatRequest

        request = ChatRequest(
            chat_id="1", new_question="q", temperature=0.1, max_new_tokens=10
        )
        params = services.resolve_params(request)
        self.assertEqual(params.temperature, 0.1)
        self.assertEqual(params.max_new_tokens, 10)


class SecurityTests(unittest.TestCase):
    """``X-Internal-Token`` trên ``/chat`` và ``/chat/batch``."""

    TOKEN = "unit-test-token"

    @staticmethod
    def _client():
        return TestClient(app)

    def test_endpoints_are_reachable_without_token_when_disabled(self):
        with mock.patch.object(settings, "INTERNAL_API_TOKEN", ""):
            client = self._client()
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/ready").status_code, 200)

    def test_ready_reports_auth_required(self):
        with mock.patch.object(settings, "INTERNAL_API_TOKEN", self.TOKEN):
            body = self._client().get("/ready").json()
        self.assertTrue(body["auth_required"])

        with mock.patch.object(settings, "INTERNAL_API_TOKEN", ""):
            body = self._client().get("/ready").json()
        self.assertFalse(body["auth_required"])

    def test_health_and_ready_stay_open_when_enabled(self):
        """Giữ mở để monitoring/script kiểm tra sống không cần biết token."""
        with mock.patch.object(settings, "INTERNAL_API_TOKEN", self.TOKEN):
            client = self._client()
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/ready").status_code, 200)

    def test_chat_rejects_missing_token(self):
        with mock.patch.object(settings, "INTERNAL_API_TOKEN", self.TOKEN):
            response = self._client().post(
                "/chat", json={"chat_id": "1", "new_question": "q"}
            )
        self.assertEqual(response.status_code, 401)

    def test_chat_rejects_wrong_token(self):
        with mock.patch.object(settings, "INTERNAL_API_TOKEN", self.TOKEN):
            response = self._client().post(
                "/chat",
                json={"chat_id": "1", "new_question": "q"},
                headers={"X-Internal-Token": "sai-token"},
            )
        self.assertEqual(response.status_code, 401)

    def test_batch_rejects_missing_token(self):
        with mock.patch.object(settings, "INTERNAL_API_TOKEN", self.TOKEN):
            response = self._client().post(
                "/chat/batch", json={"items": [{"chat_id": "1", "new_question": "q"}]}
            )
        self.assertEqual(response.status_code, 401)

    def test_chat_accepts_correct_token(self):
        with mock.patch.object(settings, "INTERNAL_API_TOKEN", self.TOKEN):
            with mock.patch.object(settings, "RAG_ENABLED", False):
                response = self._client().post(
                    "/chat",
                    json={"chat_id": "1", "new_question": "q"},
                    headers={"X-Internal-Token": self.TOKEN},
                )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["answer"])


class InputLimitTests(unittest.TestCase):
    """Trần kích thước đầu vào chống prompt khổng lồ."""

    def test_question_longer_than_limit_is_rejected(self):
        from schemas import ChatRequest

        with self.assertRaises(Exception):
            ChatRequest(
                chat_id="1",
                new_question="x" * (settings.MAX_QUESTION_LENGTH + 1),
            )

    def test_question_at_limit_is_accepted(self):
        from schemas import ChatRequest

        request = ChatRequest(
            chat_id="1", new_question="x" * settings.MAX_QUESTION_LENGTH
        )
        self.assertEqual(len(request.new_question), settings.MAX_QUESTION_LENGTH)

    def test_history_longer_than_limit_is_rejected(self):
        from schemas import ChatRequest

        with self.assertRaises(Exception):
            ChatRequest(
                chat_id="1",
                new_question="q",
                history=[
                    {"role": "user", "content": "a"}
                    for _ in range(settings.MAX_HISTORY_MESSAGES + 1)
                ],
            )

    def test_message_content_longer_than_limit_is_rejected(self):
        from schemas import ChatMessage

        with self.assertRaises(Exception):
            ChatMessage(
                role="user", content="x" * (settings.MAX_MESSAGE_LENGTH + 1)
            )

    def test_defaults_are_sane(self):
        """Các trần phải lớn hơn giá trị denso_backend gửi sang, nếu không sẽ 422."""
        self.assertGreater(settings.MAX_HISTORY_MESSAGES, 100)
        self.assertGreaterEqual(settings.MAX_QUESTION_LENGTH, 8000)


class SafeTableTests(unittest.TestCase):
    """``RAG_CHUNK_TABLE`` được nội suy vào SQL nên phải qua allow-list."""

    def test_accepts_plain_and_schema_qualified_names(self):
        from database.controllers import _safe_table

        self.assertEqual(_safe_table("rag_chunkdata"), "rag_chunkdata")
        self.assertEqual(_safe_table("public.rag_chunkdata"), "public.rag_chunkdata")

    def test_rejects_injection_attempts(self):
        from database.controllers import _safe_table

        for bad in (
            "rag_chunkdata; DROP TABLE core_users",
            "rag_chunkdata--",
            "1abc",
            "",
            "a.b.c",
            'rag_chunkdata"',
            "rag chunkdata",
        ):
            with self.assertRaises(ValueError, msg=f"phải từ chối: {bad!r}"):
                _safe_table(bad)


class EndEventCarriesContextsTests(ApiTestCase):
    """Event ``end`` của luồng SSE phải mang ``contexts``.

    Client streaming cần nguồn tham khảo ngay khi sinh xong; nếu thiếu trường này
    thì UI buộc phải gọi lại API không-stream (tốn thêm một lượt LLM) hoặc đơn
    giản là không hiển thị được nguồn.
    """

    def test_contexts_are_populated_when_rag_hits(self):
        """RAG có kết quả -> ``contexts`` trong event ``end`` phải có dữ liệu."""
        fake_context = "- đoạn tài liệu A\n- đoạn tài liệu B"

        async def fake_retrieve(questions, top_k):
            return [fake_context for _ in questions]

        with mock.patch.object(services, "retrieve_contexts", side_effect=fake_retrieve):
            response = self.client.post(
                "/chat",
                json={"chat_id": "1", "new_question": "q", "stream": True},
            )

        events = _parse_sse(response.text)
        end = [e for e in events if e.get("type") == "end"]
        self.assertTrue(end, f"không thấy event end: {events}")
        self.assertIn("contexts", end[0], "event end phải có trường contexts")
        self.assertEqual(end[0]["contexts"], [fake_context])

    def test_end_event_contexts_empty_when_no_rag_hit(self):
        with mock.patch.object(settings, "RAG_ENABLED", False):
            response = self.client.post(
                "/chat",
                json={"chat_id": "1", "new_question": "q", "stream": True},
            )

        events = _parse_sse(response.text)
        end = [e for e in events if e.get("type") == "end"]
        self.assertTrue(end)
        # Không có context -> vẫn phải có key (giá trị rỗng) để client không
        # phải phân biệt "thiếu trường" với "không có kết quả".
        self.assertEqual(end[0].get("contexts"), [])

    def test_sse_media_type_declares_utf8(self):
        """Content-Type phải khai báo charset=utf-8 (xem bug mojibake tiếng Việt)."""
        response = self.client.post(
            "/chat",
            json={"chat_id": "1", "new_question": "q", "stream": True},
        )
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn("charset=utf-8", response.headers["content-type"].lower())


if __name__ == "__main__":
    unittest.main()
