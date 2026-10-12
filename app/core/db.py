from contextlib import contextmanager
from datetime import datetime

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


def db_now() -> datetime:
    """증분 적재 워터마크(마지막 처리 시각)는 반드시 이 함수로 구해야 한다 — Python
    프로세스가 돌아가는 호스트와 Oracle 서버의 시계가 어긋나 있으면(2026-10-12, 이
    개발 PC에서 실측: Python은 KST, Oracle 컨테이너는 UTC로 9시간 차이 확인), 워터마크를
    Python의 datetime.now()로 기록하고 비교는 Oracle 쪽 타임스탬프 컬럼과 하는 식이면
    막 생긴 신규 레코드를 "이미 지난 것"으로 오판해 계속 놓칠 수 있다. 항상 Oracle
    자신의 SYSDATE를 기준으로 삼으면 두 시계가 얼마나 다르든(또는 나중에 달라지든)
    항상 안전하다."""
    return fetch_all("SELECT SYSDATE AS NOW_DTM FROM DUAL")[0]["NOW_DTM"]
