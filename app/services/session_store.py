MAX_TURNS = 10

_sessions: dict[str, list[dict]] = {}
_last_sources: dict[str, list[dict]] = {}


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
