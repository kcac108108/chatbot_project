import uuid

from qdrant_client.http import models as qmodels

from app.core.config import CHAT_TOP_K
from app.core.openai_client import chat_parse, embed_texts
from app.core.vectordb import search as vector_search
from app.models.schemas import ChatResponse, FilterExtraction, GroundedAnswer, QueryAnalysis, SourceItem
from app.services import session_store

QUERY_ANALYSIS_PROMPT = """너는 검색 전처리기다. 아래 대화 이력을 참고해 현재 질문을 분석하라.

1. standalone_question: 현재 질문이 "그 기관", "거기", "수수료만" 처럼 대명사나 생략된 맥락을 포함하면,
   대화 이력에서 언급된 구체적인 기관명·레코드를 반드시 포함시켜 대화 이력 없이도 이해되는
   독립형 질문으로 다시 써라. 예: "수수료만 알려줘" + 직전 대화가 "OO기관" 얘기였다면
   "OO기관의 수수료는 얼마인가요?"로 써라. 대상을 절대 생략하지 마라.
   맥락 의존이 없는 질문이면 원문 그대로 둔다.
2. filters: 현재 질문(과 필요 시 대화 이력)에서 검색 필터로 쓸 값을 추출해라.
   해당 정보가 명시적으로 없으면 반드시 null로 둬라. 추측하거나 지어내지 마라."""

SYSTEM_PROMPT = """너는 교부기관 이력관리시스템 챗봇이다.
아래 [자료] 목록만 근거로 답변하라. [자료]에 없는 내용은 답변에 포함하지 마라.
대화 이력(이전 turn)은 맥락 파악에만 참고하고, 사실적 근거로 인용하지 마라 — 숫자·날짜 등
구체적 사실은 반드시 이번 턴의 [자료]에 있는 값만 사용하라. 이전 답변과 이번 [자료]의 값이
다르면 이번 [자료]를 우선하라.
[자료]에서 질문과 관련된 근거를 찾지 못하면 "관련 이력을 찾지 못했습니다."라고만 답하라.
[자료]에 나온 용어(고유명사·수치·전문용어)는 바꾸지 말고 그대로 사용하되, 문장 어미까지
원문 그대로 베낄 필요는 없다. 새로운 사실을 만들어내지는 마라.
같은 내용을 다른 표현으로 반복하지 마라.

답변 형식:
- 질문이 특정 사실 하나(수수료, 담당자, 연도 등)만 물으면, "항목: 값" 같은 딱딱한 표기 대신
  자연스러운 문장 한두 개로 간결히 답하라. 예: "수수료는 230000입니다."
- 질문이 "이력 알려줘"처럼 여러 항목을 아우르면, 모든 필드를 한 문단에 욱여넣지 말고
  아래처럼 정리해서 답하라:
  첫 줄에 핵심 요약 한 문장. 그다음 줄부터 관련 있는 세부 항목만 "- 항목명: 내용" 형식으로
  줄바꿈해 나열하라. [자료]에 없거나 질문과 무관한 항목은 적지 마라.
  각 항목의 내용은 완전한 문장으로 늘어놓지 말고 핵심만 간결한 구·명사형으로 정리하라
  (예: "제공 방식은 암호화된 전자파일 형태로 전송되었다." → "암호화된 전자파일 전송").
  모든 항목이 "~다"로 끝나는 문장을 반복해 나열하지 마라.
- 여러 건(예: 신청이력 여러 개)이 근거로 쓰였으면 건별로 묶어서 나열하라.

답변에 실제로 사용한 자료 번호를 used_indices에 모두 표기하라. 근거가 없으면 used_indices는 빈 배열로 둬라."""


_NULLISH = {"null", "none", "n/a", "미상", ""}


def _build_filter(extracted: FilterExtraction) -> qmodels.Filter | None:
    conditions = []
    for field, value in extracted.model_dump(exclude_none=True).items():
        # 구조화 출력이 값 없음을 실제 null 대신 문자열 "null" 등으로 채우는 경우 방어
        if isinstance(value, str) and value.strip().lower() in _NULLISH:
            continue
        conditions.append(qmodels.FieldCondition(key=field, match=qmodels.MatchValue(value=value)))
    return qmodels.Filter(must=conditions) if conditions else None


def _analyze_query(question: str, history: list[dict]) -> QueryAnalysis:
    messages = [{"role": "system", "content": QUERY_ANALYSIS_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})
    return chat_parse(messages, QueryAnalysis)


def _search(analysis: QueryAnalysis, carried_over: list[dict]) -> list[dict]:
    vector = embed_texts([analysis.standalone_question])[0]
    query_filter = _build_filter(analysis.filters)

    results = vector_search(vector, query_filter, CHAT_TOP_K)
    if not results and query_filter is not None:
        # 필터 오추출로 결과가 0건이면 필터 없이 재검색 (질문 자체가 무관한 경우와 구분은 LLM 프롬프트가 처리)
        results = vector_search(vector, None, CHAT_TOP_K)

    # 직전 턴에서 실제로 인용된 근거는 검색이 삼천포로 빠져도 항상 후보에 남겨둔다
    # (예: "그것만 알려줘" 류 후속 질문이 검색 자체를 엉뚱한 곳으로 보내는 경우의 안전장치)
    merged = list(carried_over)
    seen = {(p["데이터유형"], p["원본ID"]) for p in merged}
    for r in results:
        key = (r.payload["데이터유형"], r.payload["원본ID"])
        if key not in seen:
            seen.add(key)
            merged.append(r.payload)
    return merged


def ask(question: str, session_id: str | None) -> ChatResponse:
    session_id = session_id or str(uuid.uuid4())
    history = session_store.get_history(session_id)
    carried_over = session_store.get_last_sources(session_id)
    analysis = _analyze_query(question, history)
    results = _search(analysis, carried_over)

    context_blocks = [f"[자료 {i + 1}] {payload['text']}" for i, payload in enumerate(results)]
    context_text = "\n\n".join(context_blocks) if context_blocks else "(검색된 자료 없음)"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": f"[자료]\n{context_text}\n\n[질문]\n{question}"})

    grounded = chat_parse(messages, GroundedAnswer, temperature=0.2)

    sources = []
    used_payloads = []
    seen = set()
    for idx in grounded.used_indices:
        if not (1 <= idx <= len(results)):
            continue
        payload = results[idx - 1]
        key = (payload["데이터유형"], payload["원본ID"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(SourceItem(데이터유형=payload["데이터유형"], 원본ID=payload["원본ID"], 기관명=payload.get("기관명")))
        used_payloads.append(payload)

    session_store.append_turn(session_id, question, grounded.answer)
    session_store.set_last_sources(session_id, used_payloads)
    return ChatResponse(session_id=session_id, answer=grounded.answer, sources=sources)
