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
