from contextlib import contextmanager

import oracledb

from app.core.config import ORACLE_DSN, ORACLE_PASSWORD, ORACLE_USER


@contextmanager
def get_connection():
    conn = oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN)
    try:
        yield conn
    finally:
        conn.close()


def fetch_all(sql: str, params: dict | None = None) -> list[dict]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params or {})
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def execute(sql: str, params: dict | None = None) -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params or {})
        conn.commit()
