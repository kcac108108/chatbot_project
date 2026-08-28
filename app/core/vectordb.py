from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import QDRANT_COLLECTION, QDRANT_HOST, QDRANT_PORT

EMBEDDING_DIM = 3072  # text-embedding-3-large 기본 차원

_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def get_client() -> QdrantClient:
    return _client


def ensure_collection() -> None:
    if _client.collection_exists(QDRANT_COLLECTION):
        return
    _client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=qmodels.VectorParams(size=EMBEDDING_DIM, distance=qmodels.Distance.COSINE),
    )


def upsert_points(points: list[qmodels.PointStruct]) -> None:
    _client.upsert(collection_name=QDRANT_COLLECTION, points=points)


def search(
    vector: list[float], query_filter: qmodels.Filter | None, limit: int
) -> list[qmodels.ScoredPoint]:
    return _client.query_points(
        collection_name=QDRANT_COLLECTION, query=vector, query_filter=query_filter, limit=limit
    ).points


def scan_text_contains(keyword: str, limit: int) -> list[dict]:
    """벡터 유사도가 아니라 본문에 특정 단어가 실제로 포함됐는지 전수 스캔한다.
    한글은 조사·복합어 때문에 Qdrant 내장 전문검색 토크나이저로는 매칭이 불안정해,
    지금 규모(수백~천 단위 청크)에서는 파이썬 쪽 부분 문자열 검사가 더 안정적이다.
    띄어쓰기 유무는 무시하고 비교한다 — 질문에 "자료 제공"(띄어쓰기 있음)과 "자료제공"
    (띄어쓰기 없음)처럼 같은 뜻인데 표기만 다른 경우, 본문의 실제 띄어쓰기와 우연히
    안 맞으면 매칭이 통째로 실패하는 문제가 재현됨."""
    normalized_keyword = "".join(keyword.split())
    matches: list[dict] = []
    offset = None
    while True:
        points, offset = _client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            text = p.payload.get("text", "")
            if normalized_keyword in "".join(text.split()):
                matches.append(p.payload)
                if len(matches) >= limit:
                    return matches
        if offset is None:
            break
    return matches
