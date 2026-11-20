import re

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import QDRANT_COLLECTION, QDRANT_HOST, QDRANT_PORT

EMBEDDING_DIM = 3072  # text-embedding-3-large 기본 차원


# 키워드를 "영문·숫자 연속 구간은 하나로 묶고, 한글은 음절 하나하나를, 그 외 문자(기호
# 등)는 한 글자씩" 단위로 쪼갠다 — 공백 자체는 세 패턴 어디에도 안 걸려 findall이 자동으로
# 건너뛴다. 이 단위들 사이에만 선택적 공백(\s*)을 허용하고 단위 "안"은 절대 안 쪼갠다.
# 그래서 "SSL" 같은 영문 약어는 항상 원자적으로 유지돼(2026-10-16에 막은 "KORS"+"SLUT"→
# "SSL" 우연한 교차매칭 위험이 재발하지 않음) 안전하고, 한글 복합명사는 음절 사이
# 공백을 자유롭게 허용해 "국가항목"↔"국가 항목" 표기 차이를 흡수한다(2026-11-05 사용자
# 제보: "국가항목을 요청한 이력"은 못 찾는데 "국가 항목을"은 찾아짐 — 순수 한글만
# 대상으로 1차 수정했다가, "SSL인증서"처럼 영문+한글이 섞인 복합어는 여전히 못 찾는
# 걸 재발견해 이 방식으로 일반화함). 기존에 키워드 자체에 이미 공백이 있던 경우
# ("국가 항목"으로 직접 입력)도 같은 토큰화를 거치므로 동일한 패턴으로 수렴한다.
# `\S`(공백 아닌 모든 문자) 폴백을 넣어 기호가 섞인 키워드("A/S" 등)에서도 그 기호가
# 토큰에서 통째로 누락돼 패턴이 아예 안 맞는 일이 없게 한다(1차 구현에서 발견해 수정).
_KEYWORD_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[가-힣]|\S")


def keyword_matches(keyword: str, *texts: str) -> bool:
    """키워드가 텍스트 안에 (띄어쓰기 차이는 무시하고) 실제로 포함되는지 확인한다.
    예전엔 haystack 전체의 공백을 전부 지우고 비교했는데, 그러면 전혀 무관한 단어들이
    우연히 이어 붙어 키워드를 만들어내는 거짓 매칭이 생긴다(2026-10-16 실측 확인:
    "MICHAEL KORS"+"SLUT" 브랜드 나열이 공백 제거 시 "...KORSSLUT..."가 되어 'SSL'을
    우연히 포함, 완전히 무관한 문서가 "SSL 관련 자료"로 지어내짐). 대신 키워드 자신을
    "영문·숫자 연속 구간"과 "한글 음절" 단위로 쪼개 그 사이에만 선택적 공백을 허용한다
    — 영문 약어("SSL")는 내부가 절대 안 쪼개지므로 단어 경계를 넘나드는 우연한 매칭
    위험이 없고, 한글은 음절 단위라 "국가항목"/"국가 항목" 표기 차이를 모두 흡수한다."""
    tokens = _KEYWORD_TOKEN_RE.findall(keyword)
    if not tokens:
        return False
    # 토큰 사이 공백 허용을 "같은 줄(스페이스/탭)"로만 한정 — 줄바꿈(\n)까지 허용하면
    # 완전히 다른 줄(다른 필드·다른 문장)의 끝 음절과 다음 줄의 시작 음절이 우연히
    # 이어져 거짓 매칭이 생긴다(2026-11-10 실측 확인: "0원\n유예기간"의 "원"+줄바꿈+
    # "유"가 "원유" 키워드와 매칭됨 — 완전히 무관한 "원"(통화단위)과 "유"(유예)가
    # 전혀 다른 두 단어의 일부인데도 합쳐짐). 2글자 한글 단어일수록 음절 각각이
    # 흔해서 문서 전체에서 이런 우연한 인접이 생길 확률이 높다. "자료 제공"/
    # "자료제공"처럼 같은 줄 안에서 띄어쓰기만 다른 경우는 그대로 허용된다.
    pattern_str = r"[ \t]*".join(re.escape(t) for t in tokens)
    # 영문·숫자 약어("DDP", "CVD" 등)는 완전히 무관한 더 긴 영숫자 토큰 안에 그냥
    # 부분문자열로 끼어있을 수 있다(2026-11-11 실측 확인: "DDP"가 국채 등기사항증명서의
    # 발급확인번호 "OAVX-MDDP-ZYW2"에 우연히 포함돼, 실제로는 무역조건(DDP, Delivered
    # Duty Paid)과 전혀 무관한 문서를 답변 LLM이 "DDP 조건이 언급된다"고 확신에 차서
    # 지어내는 걸로 이어짐). 한글은 Python의 \w 판정에 포함돼 일반적인 \b 경계를 못 쓰므로
    # (그러면 "SSL인증서"처럼 영문 뒤에 한글이 바로 붙는 정상 케이스까지 막힘), 매칭의
    # 시작/끝이 영문·숫자 토큰일 때만 "그 앞/뒤에 영문·숫자가 더 이어지지 않는지"를
    # 직접 확인하는 lookaround를 건다 — 한글·공백·기호가 앞뒤에 오는 건 전혀 안 막는다.
    if tokens[0].isascii() and tokens[0].isalnum():
        pattern_str = r"(?<![A-Za-z0-9])" + pattern_str
    if tokens[-1].isascii() and tokens[-1].isalnum():
        pattern_str = pattern_str + r"(?![A-Za-z0-9])"
    # 영문 키워드는 대소문자를 구분하면 안 된다 — "SSL"로 쓰면 찾아지는데 "ssl"/"Ssl"로
    # 쓰면 0~1건만 찾아지는 버그가 실측 확인됨(2026-11-12, "빠른 검색" 2라운드 테스트 —
    # 대문자 SSL=4건, 소문자 ssl=1건, 혼합 Ssl=0건으로 전부 다르게 나옴). 문서 작성자마다
    # 대소문자 표기가 제각각일 수 있어(SSL/ssl/Ssl 전부 실제로 등장) 사용자가 어떤
    # 대소문자로 입력하든 전부 찾아야 한다. 한글(대소문자 구분 없음)·숫자엔 영향 없음.
    pattern = re.compile(pattern_str, re.IGNORECASE)
    return any(pattern.search(t) for t in texts if t)


# 청크(text)의 첫 줄은 `chunk_builder.py`가 자동 생성하는 머리말(`[데이터유형 ID]
# 기관명(분류, 부서명)이(가)/의 ... 신청/요청/안건.` 패턴)이다. 이 부서명(CRP_DEPT_NAME)이
# 사실은 "이 레코드"가 아니라 "이 기관ID"에 한 번 등록된 부서 버킷이라, 같은 기관ID 아래의
# 모든 레코드에 똑같이 복제돼 들어간다 — 그래서 콤마로 여러 부서가 묶인 기관(2026-11-21
# 실측 확인: 대한무역투자진흥공사 한 기관ID가 "소재부품장비팀, DX가속화팀, 글로벌공급망팀"로
# 등록돼있음, 국민건강보험공단도 한 건 있음)에서는, 실제로는 "소재부품장비팀" 건인
# 레코드도 머리말 때문에 "글로벌공급망팀"으로 검색하면 똑같이 걸려버린다(사용자 실사용
# 제보로 발견 — "대한무역투자진흥공사 2026 글로벌공급망팀 청구내역"이 소재부품장비팀 건을
# 섞어서 보여줌). 머리말을 뺀 본문(요청구분/요청내용/제목 등 레코드 자신의 진짜 내용)만
# 보면 이 교차오염이 사라진다 — 기관명 자체를 찾는 경로는 이 함수가 아니라 별도 정확매칭
# (기관명 필드 필터/`scan_field_contains`)을 쓰므로 영향 없다.
_CHUNK_HEADER_LINE_RE = re.compile(r"^\[[^\]]+\][^\n]*\n?")


def strip_chunk_header(text: str) -> str:
    """청크 본문에서 자동 생성 머리말 줄 하나만 제거한다(`_CHUNK_HEADER_LINE_RE` 참고).
    순수 콘텐츠 키워드 매칭 전용 — 저장된 payload의 `text` 자체는 건드리지 않는다."""
    return _CHUNK_HEADER_LINE_RE.sub("", text or "", count=1)

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


def field_value_exists(field: str, value: str) -> bool:
    """데이터유형과 무관하게 컬렉션 전체에서 특정 필드=값인 레코드가 하나라도
    있는지만 빠르게 확인한다(limit=1, 인덱싱된 정확매칭이라 가볍다). LLM이 질문
    문맥에서 "위원회" 같은 일반명사를 기관명으로 지어내는 경우(2026-10-20 재현 —
    존재하지 않는 기관명이 필터에 들어가면 0건→단계적 완화→결국 필터가 전부
    풀려 무관한 레코드가 섞이는 연쇄)를 걸러내는 용도."""
    points, _ = _client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=1,
        with_payload=False,
        with_vectors=False,
        scroll_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(key=field, match=qmodels.MatchValue(value=value))]
        ),
    )
    return len(points) > 0


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


def scan_text_contains(keyword: str, limit: int, data_types: set[str] | None = None) -> list[dict]:
    """벡터 유사도가 아니라 본문에 특정 단어가 실제로 포함됐는지 전수 스캔한다.
    한글은 조사·복합어 때문에 Qdrant 내장 전문검색 토크나이저로는 매칭이 불안정해,
    지금 규모(수백~천 단위 청크)에서는 파이썬 쪽 부분 문자열 검사가 더 안정적이다.
    띄어쓰기 유무는 무시하고 비교한다 — 질문에 "자료 제공"(띄어쓰기 있음)과 "자료제공"
    (띄어쓰기 없음)처럼 같은 뜻인데 표기만 다른 경우, 본문의 실제 띄어쓰기와 우연히
    안 맞으면 매칭이 통째로 실패하는 문제가 재현됨.
    첨부 청크는 원문(text)뿐 아니라 사전 생성된 요약(요약)도 같이 본다 — 원문은 표·수치
    나열이라 특정 단어가 문자 그대로는 없는데 요약에서는 그 개념으로 풀어써서(예: 원문엔
    인증서 파일 목록만 있고 "SSL 인증서"라는 표현 자체는 요약에만 등장) 원문만 보면 놓치는
    사례가 재현됨(2026-09-21).
    data_types를 주면 그 데이터유형들만 스캔 결과로 센다(scan_field_contains의 data_type과
    같은 목적) — 흔한 보일러플레이트 문구(예: 계약서 표준 보안조항 "보안점검")가 특정
    메뉴(신청이력첨부 등)에 압도적으로 많으면, 사용자가 질문에 다른 메뉴("내부자료")를
    명시했는데도 그 노이즈가 limit을 먼저 채워버려 정작 찾아야 할 소수 레코드가 아예
    scan 범위 밖으로 밀리는 문제가 재현됨(2026-10-22 — "보안점검" 전체 183건 중
    174건이 신청이력첨부, 내부자료첨부는 단 1건이라 limit=30 안에 전혀 안 들어옴)."""
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
            if data_types is not None and p.payload.get("데이터유형") not in data_types:
                continue
            # 머리말(기관명/부서명 등)은 레코드 자신의 내용이 아니라 기관ID 단위로
            # 복제된 메타데이터라 여기서 빼고 본다(`strip_chunk_header` 정의부 참고).
            text = strip_chunk_header(p.payload.get("text", ""))
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


def scan_detail_field_contains(
    labels: tuple[str, ...], value: str, limit: int, data_types: set[str] | None = None
) -> list[dict]:
    """메타데이터 최상위가 아니라 `세부항목`(라벨별 원본 컬럼값 리스트) 안의 특정 라벨
    값에 부분 문자열로 포함됐는지 본다. "제공항목"/"요청항목"처럼 콤마로 구분된 여러
    항목명이 한 값에 같이 들어있는 필드(예: "HS코드, 지역, 세관, 국가, ...")에서, 그중
    하나("지역")가 실제로 제공/요청되는 자료를 찾기 위한 것 — scan_text_contains(본문
    전체 전수 스캔)로는 그 단어가 요청목적·주요내용 같은 서술문 안에 그냥 언급만 돼도
    걸려서, 진짜 대상(필드값 자체에 그 항목이 열거된 레코드)이 노이즈에 묻힌다(2026-11-18,
    차영경 "지역 항목이 제공되는 업체" 제보 — 전체 '지역' 전수스캔은 728건이나 걸리고
    그중 다수가 '지역 수출기업 모니터링'처럼 완전히 무관한 서술문 매칭, 진짜 대상(제공항목에
    '지역'을 실제로 열거한 33건)은 전혀 못 찾음). scan_field_contains와 달리 세부항목은
    레코드마다 라벨 종류가 다를 수 있어(labels 튜플로 여러 라벨 허용 — 예: 신청이력의
    "제공항목", 심의위원회의 "요청항목"/"제공항목"), 라벨이 일치하는 항목 중 하나라도
    값이 매칭되면 그 레코드를 채택한다."""
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
            if data_types is not None and p.payload.get("데이터유형") not in data_types:
                continue
            for item in p.payload.get("세부항목") or []:
                if item.get("label") not in labels:
                    continue
                field_value = item.get("value") or ""
                if normalized_value in "".join(field_value.split()):
                    matches.append(p.payload)
                    if len(matches) >= limit:
                        return matches
                    break
        if offset is None:
            break
    return matches


def find_sources_with_multiple_attachments(
    attachment_data_type: str, min_count: int = 2, limit: int = 20
) -> list[str]:
    """"첨부파일 여러 개 달린 OO 있어?" 같은 수량 질문용(2026-11-10) — 큰 첨부가 여러
    조각(청크)으로 쪼개져 있을 수 있어 단순 청크 개수로는 "첨부 몇 개"를 못 세므로,
    원본ID별로 서로 다른 첨부파일명(첨부파일명 필드)이 몇 종류인지 센다. 전수 스캔(최대
    900건 가정)이라 비용이 있지만, 질문 자체가 코퍼스 전체에 대한 존재 질문이라 narrowing
    없이는 피할 수 없다 — 벡터 top-K로 응답하면 6개 임의 표본만 보고 "없다"고 확신에 차서
    틀리게 답하는 버그가 재현됐다(ground truth 69건인데 top-6 중엔 하나도 없어 전부
    없다고 답함)."""
    counts: dict[str, set[str]] = {}
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
            if p.payload.get("데이터유형") != attachment_data_type:
                continue
            source_id = p.payload.get("원본ID")
            filename = p.payload.get("첨부파일명")
            if not source_id or not filename:
                continue
            counts.setdefault(source_id, set()).add(filename)
        if offset is None:
            break
    matched = [sid for sid, names in counts.items() if len(names) >= min_count]
    return matched[:limit]
