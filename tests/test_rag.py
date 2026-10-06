"""
Test cho tầng truy hồi RAG.

Dùng DB giả (fake cursor/connection) nên KHÔNG cần Postgres. Riêng các test
cần tới embedding sẽ được skip nếu ``sentence-transformers`` không cài đặt.
"""

import unittest
from contextlib import contextmanager
from unittest import mock

from database import controllers


class RecordingCursor:
    """Cursor giả: ghi lại câu lệnh và trả về dữ liệu theo từng lần execute."""

    def __init__(self, rows_per_call):
        self.rows_per_call = list(rows_per_call)
        self.executed = []
        self._current_rows = []

    def execute(self, query, params=None):
        self.executed.append((query, params))
        self._current_rows = (
            self.rows_per_call.pop(0) if self.rows_per_call else []
        )

    def fetchall(self):
        return self._current_rows

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeConnection:
    """Connection giả chỉ trả về cursor đã chuẩn bị sẵn."""

    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def _patch_db(cursor):
    """Trả về context manager thay thế ``controllers.get_db`` bằng DB giả."""

    @contextmanager
    def _fake_get_db():
        yield FakeConnection(cursor)

    return mock.patch.object(controllers, "get_db", _fake_get_db)


class TestSearchByEmbeddings(unittest.TestCase):
    def test_query_uses_vector_cosine_and_top_k(self):
        cursor = RecordingCursor([[("chunk A",), ("chunk B",)]])
        with _patch_db(cursor):
            contexts = controllers.search_by_embeddings([[0.1, 0.2]], top_k=3)

        self.assertEqual(contexts, ["- chunk A\n- chunk B"])
        query, params = cursor.executed[0]
        self.assertIn("rag_chunkdata", query)
        self.assertIn("vector <=> %s::vector", query)
        self.assertIn("LIMIT %s", query)
        self.assertEqual(params, ("[0.1,0.2]", 3))

    def test_empty_result_returns_empty_string(self):
        cursor = RecordingCursor([[]])
        with _patch_db(cursor):
            contexts = controllers.search_by_embeddings([[1.0]], top_k=5)
        self.assertEqual(contexts, [""])

    def test_one_query_per_embedding(self):
        cursor = RecordingCursor([[("a",)], [("b",)]])
        with _patch_db(cursor):
            contexts = controllers.search_by_embeddings([[1.0], [2.0]], top_k=2)
        self.assertEqual(len(cursor.executed), 2)
        self.assertEqual(contexts, ["- a", "- b"])

    def test_invalid_top_k_raises(self):
        with self.assertRaises(ValueError):
            controllers.search_by_embeddings([[1.0]], top_k=0)


try:
    from embedder import engine as embedder_engine

    _EMBEDDER_AVAILABLE = True
except Exception:  # pragma: no cover - thiếu sentence-transformers
    embedder_engine = None
    _EMBEDDER_AVAILABLE = False


@unittest.skipUnless(_EMBEDDER_AVAILABLE, "cần sentence-transformers")
class TestBatchSearch(unittest.TestCase):
    def test_batch_search_encodes_once_for_all_queries(self):
        cursor = RecordingCursor([[("c1",)], [("c2",)], [("c3",)]])
        with mock.patch.object(
            embedder_engine,
            "encode_texts",
            return_value=[[0.1], [0.2], [0.3]],
        ) as encode_mock, _patch_db(cursor):
            contexts = controllers.batch_search_rag_data(["q1", "q2", "q3"], top_k=4)

        # Chỉ encode 1 lần cho cả batch (batch encode).
        self.assertEqual(encode_mock.call_count, 1)
        self.assertEqual(encode_mock.call_args.args[0], ["q1", "q2", "q3"])
        self.assertEqual(contexts, ["- c1", "- c2", "- c3"])

    def test_batch_search_empty_queries(self):
        cursor = RecordingCursor([])
        with _patch_db(cursor):
            contexts = controllers.batch_search_rag_data([], top_k=3)
        self.assertEqual(contexts, [])

    def test_search_rag_data_returns_single_context(self):
        cursor = RecordingCursor([[("only",)]])
        with mock.patch.object(
            embedder_engine, "encode_texts", return_value=[[0.5]]
        ), _patch_db(cursor):
            context = controllers.search_rag_data("q", top_k=1)
        self.assertEqual(context, "- only")


if __name__ == "__main__":
    unittest.main()
