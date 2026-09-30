MAX_TURNS = 10

_sessions: dict[str, list[dict]] = {}
_last_sources: dict[str, list[dict]] = {}
# "전체 이력" 응답 직후, 그 목록 전체(가벼운 HistoryItem이 아니라 실제 본문이 담긴
# payload)를 따로 저장해둔다 — 후속 질문("그 중 OO 관련된 것만")이 왔을 때 코드가
# 직접 텍스트 매칭으로 필터링할 수 있게 하기 위함(2026-09-30). 일반 carried-over
# 안전망(_last_sources)과 섞으면 매 턴 무제한으로 커질 위험이 있어 별도로 둔다.
_last_full_history: dict[str, tuple[str, list[dict]]] = {}


def get_history(session_id: str) -> list[dict]:
    return _sessions.get(session_id, [])


def append_turn(session_id: str, question: str, answer: str) -> None:
    history = _sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    del history[: max(0, len(history) - MAX_TURNS * 2)]


def get_last_sources(session_id: str) -> list[dict]:
    return _last_sources.get(session_id, [])


def set_last_sources(session_id: str, payloads: list[dict]) -> None:
    _last_sources[session_id] = payloads


def get_last_full_history(session_id: str) -> tuple[str, list[dict]] | None:
    return _last_full_history.get(session_id)


def set_last_full_history(session_id: str, org_name: str, payloads: list[dict]) -> None:
    _last_full_history[session_id] = (org_name, payloads)
