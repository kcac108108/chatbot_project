import uuid

from qdrant_client.http import models as qmodels

from app.core.db import fetch_all
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


def _collect_all_chunks() -> list[dict]:
    chunks: list[dict] = []
    chunks += build_menu1_chunks(fetch_all(MENU1_QUERY))
    chunks += build_menu2_chunks(fetch_all(MENU2_QUERY))
    chunks += build_menu3_chunks(fetch_all(MENU3_QUERY))
    chunks += build_menu4_chunks(fetch_all(MENU4_QUERY))
    return chunks


def run_ingest() -> int:
    ensure_collection()
    chunks = _collect_all_chunks()

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


if __name__ == "__main__":
    count = run_ingest()
    print(f"[ingest] 전체 완료: {count}개 청크 적재")
