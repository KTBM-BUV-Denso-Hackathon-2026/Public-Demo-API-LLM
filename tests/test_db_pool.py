"""
Test cho việc loại bỏ kết nối chết khi mượn từ connection pool.

Bối cảnh: ``DATABASE_URL`` trỏ tới connection POOLER của cloud Postgres
(Neon/Supabase). Pooler chủ động đóng kết nối rảnh, còn pool của ta vẫn giữ
socket đã chết. Không có kiểm tra này thì query sẽ nổ::

    psycopg2.OperationalError: SSL SYSCALL error:
    Software caused connection abort (0x00002745/10053)

Dùng pool giả nên test chạy offline, không cần Postgres.
"""

import unittest
from unittest import mock

import psycopg2

from database import config


class FakeCursor:
    """Cursor giả: ném lỗi nếu kết nối được cho là đã chết."""

    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, query, params=None):
        if not self._conn.alive:
            raise psycopg2.OperationalError(
                "SSL SYSCALL error: Software caused connection abort "
                "(0x00002745/10053)"
            )


class FakeConnection:
    def __init__(self, name, alive=True):
        self.name = name
        self.alive = alive
        self.rolled_back = 0

    @property
    def closed(self):
        return 0

    def cursor(self):
        return FakeCursor(self)

    def rollback(self):
        self.rolled_back += 1

    def get_transaction_status(self):
        # Luôn IDLE: sau khi ping thành công thì không còn transaction nào treo.
        return psycopg2.extensions.TRANSACTION_STATUS_IDLE

    def __repr__(self):  # pragma: no cover - chỉ phục vụ thông báo lỗi
        return "<FakeConnection %s alive=%s>" % (self.name, self.alive)


class FakePool:
    """Pool giả: trả về lần lượt các kết nối đã dựng sẵn."""

    def __init__(self, connections):
        self._queue = list(connections)
        self.discarded = []
        self.returned = []

    def getconn(self):
        return self._queue.pop(0)

    def putconn(self, conn, close=False):
        if close:
            self.discarded.append(conn)
        else:
            self.returned.append(conn)
            self._queue.append(conn)


class BorrowLiveConnectionTests(unittest.TestCase):
    """``_borrow_live_connection`` phải loại kết nối chết và thử lại."""

    def test_healthy_connection_is_returned_immediately(self):
        conn = FakeConnection("A", alive=True)
        pool = FakePool([conn])

        result = config._borrow_live_connection(pool, attempts=2)

        self.assertIs(result, conn)
        self.assertEqual(pool.discarded, [])

    def test_dead_connection_is_discarded_and_retry_succeeds(self):
        dead = FakeConnection("A", alive=False)
        alive = FakeConnection("B", alive=True)
        pool = FakePool([dead, alive])

        result = config._borrow_live_connection(pool, attempts=2)

        self.assertIs(result, alive)
        self.assertEqual(pool.discarded, [dead])
        # Kết nối sống KHÔNG được trả về pool ngay ở bước mượn.
        self.assertEqual(pool.returned, [])

    def test_raises_when_every_connection_is_dead(self):
        pool = FakePool(
            [FakeConnection("A", alive=False), FakeConnection("B", alive=False)]
        )

        with self.assertRaises(psycopg2.OperationalError):
            config._borrow_live_connection(pool, attempts=2)

        self.assertEqual(len(pool.discarded), 2)

    def test_single_attempt_does_not_retry(self):
        pool = FakePool([FakeConnection("A", alive=False), FakeConnection("B", alive=True)])

        with self.assertRaises(psycopg2.OperationalError):
            config._borrow_live_connection(pool, attempts=1)

        self.assertEqual(len(pool.discarded), 1)
        # Không được lấy tới kết nối thứ hai.
        self.assertEqual(len(pool._queue), 1)

    def test_attempts_zero_still_makes_one_try(self):
        """attempts<=0 không được biến thành "không thử lần nào"."""
        pool = FakePool([FakeConnection("A", alive=True)])

        result = config._borrow_live_connection(pool, attempts=0)

        self.assertIs(result, pool._queue[0] if pool._queue else result)
        self.assertTrue(result.alive)


class GetDbTests(unittest.TestCase):
    """``get_db`` phải đi qua bước kiểm tra kết nối sống."""

    def test_get_db_pings_before_use(self):
        conn = FakeConnection("A", alive=True)
        pool = FakePool([conn])

        with mock.patch.object(config, "get_pool", return_value=pool):
            with config.get_db() as borrowed:
                self.assertIs(borrowed, conn)

        # Kết nối trả về pool kèm rollback dọn transaction.
        self.assertEqual(pool.returned, [conn])
        self.assertEqual(pool.discarded, [])

    def test_get_db_transparently_replaces_dead_connection(self):
        dead = FakeConnection("A", alive=False)
        alive = FakeConnection("B", alive=True)
        pool = FakePool([dead, alive])

        with mock.patch.object(config, "get_pool", return_value=pool):
            with mock.patch.object(
                config.settings, "DB_PREPING_ATTEMPTS", 2, create=True
            ):
                with config.get_db() as borrowed:
                    self.assertIs(borrowed, alive)

        self.assertEqual(pool.discarded, [dead])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()