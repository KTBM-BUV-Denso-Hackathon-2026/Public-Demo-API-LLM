"""Test cho ``llm.client``: provider mock (chạy offline, không cần model thật)."""

import unittest
from unittest import mock

from llm import client
from llm.client import GenerationParams


class TestMockProvider(unittest.IsolatedAsyncioTestCase):
    async def test_complete_uses_context_from_prompt(self):
        messages = [
            {"role": "system", "content": "Bạn là trợ lý."},
            {
                "role": "user",
                "content": (
                    "THÔNG TIN THAM KHẢO:\n- Bước 1: kiểm tra dầu\n- Bước 2: thay lọc\n\n"
                    "CÂU HỎI: Quy trình bảo trì?"
                ),
            },
        ]

        with mock.patch.object(client.settings, "LLM_PROVIDER", "mock"):
            answer = await client.complete(messages)

        self.assertIn(client.MOCK_MARKER, answer)
        self.assertIn("Quy trình bảo trì?", answer)
        self.assertIn("Số đoạn ngữ cảnh được truy hồi: 2", answer)
        self.assertIn("Bước 1: kiểm tra dầu", answer)

    async def test_complete_without_context_says_so(self):
        with mock.patch.object(client.settings, "LLM_PROVIDER", "mock"):
            answer = await client.complete(
                [{"role": "user", "content": "Không có context"}]
            )

        self.assertIn("Không có ngữ cảnh RAG nào được truy hồi.", answer)

    async def test_stream_yields_whole_answer(self):
        with mock.patch.object(
            client.settings, "LLM_PROVIDER", "mock"
        ), mock.patch.object(client.settings, "MOCK_STREAM_DELAY_MS", 0):
            chunks = []
            async for delta in client.stream(
                [{"role": "user", "content": "Xin chào"}]
            ):
                chunks.append(delta)

        self.assertGreater(len(chunks), 1)  # phải được chia thành nhiều delta
        joined = "".join(chunks)
        self.assertIn(client.MOCK_MARKER, joined)
        self.assertIn("Xin chào", joined)

    async def test_is_mock_flag(self):
        with mock.patch.object(client.settings, "LLM_PROVIDER", "mock"):
            self.assertTrue(client.is_mock())

        with mock.patch.object(client.settings, "LLM_PROVIDER", "openai_compatible"):
            self.assertFalse(client.is_mock())


class TestGenerationParams(unittest.TestCase):
    def test_defaults_come_from_settings(self):
        params = GenerationParams()

        self.assertEqual(params.model, client.settings.LLM_MODEL)
        self.assertEqual(params.temperature, client.settings.DEFAULT_TEMPERATURE)
        self.assertEqual(params.max_new_tokens, client.settings.DEFAULT_MAX_NEW_TOKENS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
