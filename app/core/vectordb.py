import re

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import QDRANT_COLLECTION, QDRANT_HOST, QDRANT_PORT

EMBEDDING_DIM = 3072  # text-embedding-3-large 기본 차원


def keyword_matches(keyword: str, *texts: str) -> bool:
    """키워드가 텍스트 안에 (띄어쓰기 차이는 무시하고) 실제로 포함되는지 확인한다.
    키워드 자신의 공백 위치에만 선택적 공백(\\s*)을 허용하고 나머지는 원문 그대로
    검색한다 — 예전엔 haystack 전체의 공백을 전부 지우고 비교했는데, 그러면 전혀
    무관한 단어들이 우연히 이어 붙어 키워드를 만들어내는 거짓 매칭이 생긴다
    (2026-10-16 실측 확인: "MICHAEL KORS"+"SLUT" 브랜드 나열이 공백 제거 시
    "...KORSSLUT..."가 되어 'SSL'을 우연히 포함, 완전히 무관한 문서가 "SSL 관련
    자료"로 지어내짐 — 짧은 키워드일수록 이 우연한 교차 매칭 위험이 큼). 키워드
    내부에 공백이 없으면(예: "SSL") 이 함수는 사실상 평범한 리터럴 부분 문자열
    검색과 같아져 원문의 단어 경계를 그대로 존중한다."""
    parts = keyword.split()
    if not parts:
        return False
    pattern = re.compile(r"\s*".join(re.escape(p) for p in parts))
    return any(pattern.search(t) for t in texts if t)

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


def find_by_type_and_source(data_type: str, source_id: str, limit: int = 20) -> list[dict]:
    """특정 원본ID에 딸린 첨부내용 청크처럼, (데이터유형, 원본ID)로 정확히 매칭되는
    payload를 전부 가져온다 — 메인 이력 하나에 첨부 청크가 여러 개(파일이 여럿, 또는
    큰 파일이 여러 조각으로 쪼개진 경우) 있을 수 있어 리스트로 반환한다."""
    points, _ = _client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=limit,
        with_payload=True,
        with_vectors=False,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="데이터유형", match=qmodels.MatchValue(value=data_type)),
                qmodels.FieldCondition(key="원본ID", match=qmodels.MatchValue(value=source_id)),
            ]
        ),
    )
    return [p.payload for p in points]


def find_by_type_and_field(data_type: str, field: str, value: str, limit: int = 50) -> list[dict]:
    """(데이터유형, 필드=값)으로 정확히 매칭되는 payload를 전부 가져온다. 기관정보처럼
    특정 기관명으로 좁힌 레코드는 원래 건수가 적어(보통 1~4건, 중복 등록 포함해도)
    top-K 벡터 검색 상한에 걸려 잘릴 위험 없이 그냥 전부 가져와도 안전하다."""
    points, _ = _client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=limit,
        with_payload=True,
        with_vectors=False,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="데이터유형", match=qmodels.MatchValue(value=data_type)),
                qmodels.FieldCondition(key=field, match=qmodels.MatchValue(value=value)),
            ]
        ),
    )
    return [p.payload for p in points]


def find_by_type_and_fields(data_type: str, fields: dict[str, str], limit: int = 200) -> list[dict]:
    """(데이터유형, 필드1=값1 AND 필드2=값2 ...)으로 정확히 매칭되는 payload를 전부
    가져온다. 첨부 청크엔 데이터유형/원본ID/첨부파일명/요약/text 외의 메타데이터 필드가
    없어서(기관명·연도·요청자 등 전부 부모 레코드에만 있음), 첨부류 스코프에서 이런
    필드로 좁히려면 먼저 부모 레코드를 이 함수로 찾아 원본ID로 바꿔야 한다(2026-10-13,
    app/services/chat_service.py의 _build_filter 참고)."""
    if not fields:
        return []
    points, _ = _client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=limit,
        with_payload=True,
        with_vectors=False,
        scroll_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(key="데이터유형", match=qmodels.MatchValue(value=data_type))]
            + [qmodels.FieldCondition(key=field, match=qmodels.MatchValue(value=value)) for field, value in fields.items()]
        ),
    )
    return [p.payload for p in points]


def scan_text_contains(keyword: str, limit: int) -> list[dict]:
    """벡터 유사도가 아니라 본문에 특정 단어가 실제로 포함됐는지 전수 스캔한다.
    한글은 조사·복합어 때문에 Qdrant 내장 전문검색 토크나이저로는 매칭이 불안정해,
    지금 규모(수백~천 단위 청크)에서는 파이썬 쪽 부분 문자열 검사가 더 안정적이다.
    띄어쓰기 유무는 무시하고 비교한다 — 질문에 "자료 제공"(띄어쓰기 있음)과 "자료제공"
    (띄어쓰기 없음)처럼 같은 뜻인데 표기만 다른 경우, 본문의 실제 띄어쓰기와 우연히
    안 맞으면 매칭이 통째로 실패하는 문제가 재현됨.
    첨부 청크는 원문(text)뿐 아니라 사전 생성된 요약(요약)도 같이 본다 — 원문은 표·수치
    나열이라 특정 단어가 문자 그대로는 없는데 요약에서는 그 개념으로 풀어써서(예: 원문엔
    인증서 파일 목록만 있고 "SSL 인증서"라는 표현 자체는 요약에만 등장) 원문만 보면 놓치는
    사례가 재현됨(2026-09-21)."""
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
            summary = p.payload.get("요약", "") or ""
            if keyword_matches(keyword, text, summary):
                matches.append(p.payload)
                if len(matches) >= limit:
                    return matches
        if offset is None:
            break
    return matches


def scan_field_contains(field: str, value: str, limit: int, data_type: str | None = None) -> list[dict]:
    """본문 전체가 아니라 메타데이터의 특정 필드 값 안에 부분 문자열로 포함됐는지만 본다.
    요청자/제공자/담당자/결재자처럼 한 필드에 여러 사람이 콤마로 같이 저장된 경우
    (예: 담당자="김인태, 전봉석")를 잡기 위한 것 — scan_text_contains로 본문 전체를
    스캔하면, 그 이름이 이 필드가 아니라 다른 곳(예: 서명란, 참고 언급)에 흔하게 등장하는
    바람에 상한(limit)이 무관한 매치들로 먼저 채워져 정작 찾아야 할 레코드가 밀려나는
    문제가 재현됨(2026-09-21, "전봉석"이 여러 문서의 담당자 서명으로 흔히 등장). 필드
    값만 좁혀서 보면 이런 노이즈 없이 정확히 그 역할로 저장된 레코드만 찾는다.
    data_type을 주면 그 데이터유형만 스캔한다 — 안 주면(기본) 전체 스캔(기존 동작 그대로).
    같은 "노이즈가 먼저 상한을 채우는" 문제가 데이터유형 간에도 재현됨(2026-10-19, 기관명
    줄임말 검색 — "환경공단"이 "한국환경공단"의 기관정보(1건)보다 신청이력(83건)에 훨씬
    많이 등장해서, 데이터유형 필터 없이 돌리면 기관정보는 상한 안에 전혀 안 들어옴)."""
    normalized_value = "".join(value.split())
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
            if data_type is not None and p.payload.get("데이터유형") != data_type:
                continue
            field_value = p.payload.get(field) or ""
            if normalized_value in "".join(field_value.split()):
                matches.append(p.payload)
                if len(matches) >= limit:
                    return matches
        if offset is None:
            break
    return matches
