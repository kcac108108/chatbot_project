import uuid

from qdrant_client.http import models as qmodels

from app.core.config import CHAT_TOP_K
from app.core.openai_client import chat_parse, embed_texts
from app.core.vectordb import search as vector_search
from app.models.schemas import ChatResponse, FilterExtraction, GroundedAnswer, QueryAnalysis, SourceItem
from app.services import session_store

QUERY_ANALYSIS_PROMPT = """너는 검색 전처리기다. 아래 대화 이력을 참고해 현재 질문을 분석하라.

1. standalone_question: 현재 질문이 "그 기관", "거기" 같은 대명사나 생략된 맥락을 포함하면,
   대화 이력을 반영해 대화 이력 없이도 이해되는 독립형 질문으로 다시 써라.
   맥락 의존이 없는 질문이면 원문 그대로 둔다.
2. filters: 현재 질문(과 필요 시 대화 이력)에서 검색 필터로 쓸 값을 추출해라.
   해당 정보가 없으면 반드시 null로 둬라. 추측해서 채우지 마라."""

SYSTEM_PROMPT = """너는 교부기관 이력관리시스템 챗봇이다.
아래 [자료] 목록만 근거로 답변하라. [자료]에 없는 내용은 답변에 포함하지 마라.
[자료]에서 질문과 관련된 근거를 찾지 못하면 "관련 이력을 찾지 못했습니다."라고만 답하라.
[자료]에 나온 용어와 표현을 그대로 사용하고, 새로운 단어를 만들어내지 마라.
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


def _search(analysis: QueryAnalysis) -> list[qmodels.ScoredPoint]:
    vector = embed_texts([analysis.standalone_question])[0]
    query_filter = _build_filter(analysis.filters)

    results = vector_search(vector, query_filter, CHAT_TOP_K)
    if not results and query_filter is not None:
        # 필터 오추출로 결과가 0건이면 필터 없이 재검색 (질문 자체가 무관한 경우와 구분은 LLM 프롬프트가 처리)
        results = vector_search(vector, None, CHAT_TOP_K)
    return results


def ask(question: str, session_id: str | None) -> ChatResponse:
    session_id = session_id or str(uuid.uuid4())
    history = session_store.get_history(session_id)
    analysis = _analyze_query(question, history)
    results = _search(analysis)

    context_blocks = [f"[자료 {i + 1}] {r.payload['text']}" for i, r in enumerate(results)]
    context_text = "\n\n".join(context_blocks) if context_blocks else "(검색된 자료 없음)"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": f"[자료]\n{context_text}\n\n[질문]\n{question}"})

    grounded = chat_parse(messages, GroundedAnswer, temperature=0.2)

    sources = []
    seen = set()
    for idx in grounded.used_indices:
        if not (1 <= idx <= len(results)):
            continue
        payload = results[idx - 1].payload
        key = (payload["데이터유형"], payload["원본ID"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(SourceItem(데이터유형=payload["데이터유형"], 원본ID=payload["원본ID"], 기관명=payload.get("기관명")))

    session_store.append_turn(session_id, question, grounded.answer)
    return ChatResponse(session_id=session_id, answer=grounded.answer, sources=sources)
