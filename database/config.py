import psycopg2
import dotenv
import os
from contextlib import contextmanager

dotenv.load_dotenv()

@contextmanager
def get_db():
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    try:
        yield conn
    finally:
        conn.close()

