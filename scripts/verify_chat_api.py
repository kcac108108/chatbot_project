"""정의서 7장 검증기준 5번(검색 시나리오) + 6.5절(후속 질문 대응)을
실제 POST /chat 엔드포인트로 검증한다.

사전 조건: uvicorn app.main:app 서버가 떠 있어야 함.
실행: venv/Scripts/python.exe scripts/verify_chat_api.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from app.core.db import fetch_all

BASE_URL = "http://127.0.0.1:8000"
ok = True


def check(label: str, condition: bool, detail: str = ""):
    global ok
    mark = "PASS" if condition else "FAIL"
    if not condition:
        ok = False
    print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""))


def ask(question: str, session_id: str | None = None) -> dict:
    resp = httpx.post(f"{BASE_URL}/chat", json={"question": question, "session_id": session_id}, timeout=60)
    resp.raise_for_status()
    return resp.json()


print("=" * 60)
print("검색 시나리오 (정의서 7장 5번)")
print("=" * 60)

org = fetch_all("SELECT CRP_NAME FROM HISTORY_ORGANIZATION_INFO WHERE ORGANIZATION_ID = 'ORG0000011'")[0]["CRP_NAME"]
r = ask(f"{org} 관련 이력 알려줘")
check(f"기관명 검색 — '{org}'", org in r["answer"] and any(s["기관명"] == org for s in r["sources"]),
      f"answer={r['answer'][:60]}...")

year_row = fetch_all(
    "SELECT SUBSTR(REQ_DTM,1,4) AS YR FROM HISTORY_ORGANIZATION_DETAIL WHERE REQ_DTM IS NOT NULL FETCH FIRST 1 ROWS ONLY"
)[0]
year = year_row["YR"]
r = ask(f"{year}년에 신청된 이력 알려줘")
check(f"연도 필터 검색 — {year}년", len(r["sources"]) > 0, f"sources={r['sources']}")

result_row = fetch_all(
    "SELECT DISCUSSION_RESULT FROM HISTORY_DISCUSSION_COMMITTEE WHERE DISCUSSION_RESULT IS NOT NULL FETCH FIRST 1 ROWS ONLY"
)[0]
result_val = "".join(result_row["DISCUSSION_RESULT"].split())
r = ask(f"심의결과가 {result_val}인 안건 알려줘")
check(f"심의결과 필터 검색 — {result_val}", len(r["sources"]) > 0, f"sources={r['sources']}")

person = fetch_all("SELECT USERNM FROM HISTORY_INTERNAL_INFO WHERE USERNM IS NOT NULL FETCH FIRST 1 ROWS ONLY")[0][
    "USERNM"
]
r = ask(f"{person}이 담당한 자료 알려줘")
check(f"담당자 검색 — '{person}'", person in r["answer"], f"answer={r['answer'][:60]}...")

requester = fetch_all(
    "SELECT REQ_PERSON_NAME FROM HISTORY_EXTERNAL_REQUEST WHERE REQ_PERSON_NAME IS NOT NULL FETCH FIRST 1 ROWS ONLY"
)[0]["REQ_PERSON_NAME"]
r = ask(f"{requester}이 요청한 외부요청 알려줘")
check(f"요청자 검색 — '{requester}'", requester in r["answer"], f"answer={r['answer'][:60]}...")

r = ask("존재하지않는가상기관XYZ99999의 심의 이력 알려줘")
check("존재하지 않는 건 질문 — '없음' 응답", "찾지 못했" in r["answer"] and len(r["sources"]) == 0,
      f"answer={r['answer']}")

print()
print("=" * 60)
print("후속 질문 대응 (정의서 6.5절 세션 유지)")
print("=" * 60)

r1 = ask(f"{org} 관련 이력 알려줘")
sid = r1["session_id"]
r2 = ask("그 기관 사업명이 뭐야?", session_id=sid)
check("대명사 후속 질문이 세션 맥락으로 해결됨", org in r2["answer"] and r2["session_id"] == sid,
      f"answer={r2['answer']}")

print()
print("=" * 60)
print("전체 결과:", "PASS" if ok else "FAIL — 위 FAIL 항목 확인 필요")
print("=" * 60)
sys.exit(0 if ok else 1)
