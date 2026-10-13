import random
import time

MAX_TURNS = 10

# 세션이 전부 메모리에만 저장되고 만료 로직이 없어서, 서버를 오래 안 끄고 운영하면
# session_id가 계속 쌓이기만 하는 구조였음(2026-10-13 배포 준비 점검 중 발견 — 각 세션
# 자체는 MAX_TURNS로 크기가 제한되지만 세션 "개수"는 무한정 증가). 접속할 때마다 새
# session_id가 생기므로 오래 방치하면 서서히 메모리가 늘어남. 백그라운드 스레드/스케줄러
# 없이 가볍게 처리하도록, 호출마다 낮은 확률로 만료된 세션을 청소하는 방식 채택.
SESSION_TTL_SECONDS = 24 * 60 * 60
_CLEANUP_PROBABILITY = 0.01

_sessions: dict[str, list[dict]] = {}
_last_sources: dict[str, list[dict]] = {}
_last_access: dict[str, float] = {}


def _touch(session_id: str) -> None:
    _last_access[session_id] = time.time()
    if random.random() < _CLEANUP_PROBABILITY:
        _cleanup_expired()


def _cleanup_expired() -> None:
    cutoff = time.time() - SESSION_TTL_SECONDS
    expired = [sid for sid, accessed_at in _last_access.items() if accessed_at < cutoff]
    for sid in expired:
        _sessions.pop(sid, None)
        _last_sources.pop(sid, None)
        _last_access.pop(sid, None)


def get_history(session_id: str) -> list[dict]:
    _touch(session_id)
    return _sessions.get(session_id, [])


def append_turn(session_id: str, question: str, answer: str) -> None:
    _touch(session_id)
    history = _sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    del history[: max(0, len(history) - MAX_TURNS * 2)]


def get_last_sources(session_id: str) -> list[dict]:
    _touch(session_id)
    return _last_sources.get(session_id, [])


def set_last_sources(session_id: str, payloads: list[dict]) -> None:
    _touch(session_id)
    _last_sources[session_id] = payloads
