"""
Test cho ``embedder.engine`` bằng model giả.

Không tải model thật (không cần mạng): chỉ kiểm tra logic bọc ngoài —
cache model, chuyển numpy -> list, và kiểm tra số chiều.
"""

import unittest
from unittest import mock

import numpy as np

import settings
from embedder import engine


class FakeModel:
    """Model giả trả về vector 0 với số chiều cho trước."""

    def __init__(self, dimension):
        self.dimension = dimension

    def encode(self, texts, batch_size=None, normalize_embeddings=True, convert_to_numpy=True):
        return np.zeros((len(texts), self.dimension), dtype="float32")

    def get_sentence_embedding_dimension(self):
        return self.dimension


class TestEncodeTexts(unittest.TestCase):
    def test_empty_input_returns_empty(self):
        self.assertEqual(engine.encode_texts([]), [])

    def test_batch_encode_returns_plain_lists(self):
        # FakeModel trả 4 chiều nên phải khai báo EMBEDDING_DIM=4 cho khớp.
        with mock.patch.object(settings, "EMBEDDING_DIM", 4), mock.patch.object(
            engine, "get_model", return_value=FakeModel(4)
        ):
            vectors = engine.encode_texts(["a", "b", "c"])

        self.assertEqual(len(vectors), 3)
        self.assertEqual(len(vectors[0]), 4)
        self.assertIsInstance(vectors[0], list)
        self.assertIsInstance(vectors[0][0], float)

    def test_strict_dimension_mismatch_raises(self):
        original_dim = settings.EMBEDDING_DIM
        original_strict = settings.STRICT_EMBEDDING_DIM
        settings.EMBEDDING_DIM = 8
        settings.STRICT_EMBEDDING_DIM = True
        try:
            with mock.patch.object(engine, "get_model", return_value=FakeModel(4)):
                with self.assertRaises(RuntimeError):
                    engine.encode_texts(["a"])
        finally:
            settings.EMBEDDING_DIM = original_dim
            settings.STRICT_EMBEDDING_DIM = original_strict

    def test_auto_dimension_does_not_raise(self):
        """EMBEDDING_DIM=0 nghĩa là tự động -> không kiểm tra/không raise."""
        original_dim = settings.EMBEDDING_DIM
        settings.EMBEDDING_DIM = 0
        try:
            with mock.patch.object(engine, "get_model", return_value=FakeModel(4)):
                vectors = engine.encode_texts(["a"])
            self.assertEqual(len(vectors[0]), 4)
        finally:
            settings.EMBEDDING_DIM = original_dim


class TestEmbeddingDefaults(unittest.TestCase):
    """Cấu hình mặc định phải khớp cột vector(768) trong DB."""

    def test_default_dimension_matches_database(self):
        self.assertEqual(settings.EMBEDDING_DIM, 768)

    def test_dimension_check_is_strict_by_default(self):
        self.assertTrue(settings.STRICT_EMBEDDING_DIM)


class TestModelCache(unittest.TestCase):
    def tearDown(self):
        engine.reset_cache()

    def test_model_loaded_once(self):
        engine.reset_cache()
        fake_model = FakeModel(3)
        constructor = mock.MagicMock(return_value=fake_model)
        with mock.patch.object(engine, "SentenceTransformer", constructor):
            first = engine.get_model("demo-model")
            second = engine.get_model("demo-model")

        self.assertIs(first, second)
        self.assertEqual(constructor.call_count, 1)

    def test_dimension_helper(self):
        with mock.patch.object(engine, "get_model", return_value=FakeModel(7)):
            self.assertEqual(engine.get_model_dimension("x"), 7)


if __name__ == "__main__":
    unittest.main()
