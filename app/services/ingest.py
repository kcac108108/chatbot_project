import sys
import uuid
from datetime import datetime

from qdrant_client.http import models as qmodels

from app.core.db import execute, fetch_all
from app.core.openai_client import embed_texts
from app.core.vectordb import ensure_collection, upsert_points
from app.services.chunk_builder import (
    build_menu1_chunks,
    build_menu2_chunks,
    build_menu3_chunks,
    build_menu4_chunks,
)
from app.services.queries import MENU1_QUERY, MENU2_QUERY, MENU3_QUERY, MENU4_QUERY

BATCH_SIZE = 100


def _point_id(data_type: str, source_id) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{data_type}:{source_id}"))


def _collect_all_chunks(since: datetime | None = None) -> list[dict]:
    params = {"since": since}
    chunks: list[dict] = []
    chunks += build_menu1_chunks(fetch_all(MENU1_QUERY, params))
    chunks += build_menu2_chunks(fetch_all(MENU2_QUERY, params))
    chunks += build_menu3_chunks(fetch_all(MENU3_QUERY, params))
    chunks += build_menu4_chunks(fetch_all(MENU4_QUERY, params))
    return chunks


def run_ingest(since: datetime | None = None) -> int:
    """전체(또는 since 이후 변경분) 재적재. since가 None이면 항상 전체 재적재이며,
    이 경우 CHATBOT_INGEST_LOG 기록은 남기지 않는다(증분 기준점은 run_incremental_ingest
    전용) — 전체 재적재는 실데이터 초기 반입처럼 별도 목적으로 쓰이는 경우가 많아
    증분 스케줄과 뒤섞이면 안 된다."""
    ensure_collection()
    chunks = _collect_all_chunks(since)

    total = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        vectors = embed_texts([c["text"] for c in batch])
        points = [
            qmodels.PointStruct(
                id=_point_id(c["metadata"]["데이터유형"], c["metadata"]["원본ID"]),
                vector=vector,
                payload={"text": c["text"], **c["metadata"]},
            )
            for c, vector in zip(batch, vectors)
        ]
        upsert_points(points)
        total += len(points)
        print(f"[ingest] {total}/{len(chunks)} 청크 적재 완료")

    return total


def _get_last_ingest_at() -> datetime | None:
    rows = fetch_all("SELECT MAX(RUN_AT) AS LAST_RUN_AT FROM CHATBOT_INGEST_LOG")
    return rows[0]["LAST_RUN_AT"] if rows else None


def _record_ingest_run(run_at: datetime, chunk_count: int) -> None:
    execute(
        "INSERT INTO CHATBOT_INGEST_LOG (RUN_AT, CHUNK_COUNT) VALUES (:run_at, :chunk_count)",
        {"run_at": run_at, "chunk_count": chunk_count},
    )


def run_incremental_ingest() -> int:
    """CHATBOT_INGEST_LOG에 기록된 마지막 실행 시각 이후 신규·수정된 레코드만 골라
    재적재한다(신규 등록/수정 둘 다 IN_DTM·CR_DTM 중 하나는 갱신된다고 보고 둘 다 확인).
    Qdrant 포인트ID가 (데이터유형, 원본ID) 기준 결정론적이라, 기존 레코드가 다시
    걸려도 중복 없이 덮어쓰기(upsert)만 된다. 실행 시각은 조회 시작 "전" 시점을
    기준으로 기록해, 조회 도중 새로 들어온 레코드가 다음 실행에서도 다시 잡히게 한다."""
    run_started_at = datetime.now()
    since = _get_last_ingest_at()
    count = run_ingest(since)
    _record_ingest_run(run_started_at, count)
    return count


if __name__ == "__main__":
    if "--incremental" in sys.argv:
        count = run_incremental_ingest()
        print(f"[ingest] 증분 적재 완료: {count}개 청크 적재")
    else:
        count = run_ingest()
        print(f"[ingest] 전체 완료: {count}개 청크 적재")
