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


if __name__ == "__main__":
    unittest.main()
