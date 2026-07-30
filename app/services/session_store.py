MAX_TURNS = 10

_sessions: dict[str, list[dict]] = {}


def get_history(session_id: str) -> list[dict]:
    return _sessions.get(session_id, [])


def append_turn(session_id: str, question: str, answer: str) -> None:
    history = _sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    del history[: max(0, len(history) - MAX_TURNS * 2)]
