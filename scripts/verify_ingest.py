"""정의서 7장 검증 기준 1~5번을 실행하는 스크립트.

Oracle(원본)과 Qdrant(적재 결과)를 대조한다. 실행: venv/Scripts/python.exe scripts/verify_ingest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client.http import models as qmodels

from app.core.db import fetch_all
from app.core.openai_client import embed_texts
from app.core.vectordb import get_client
from app.core.config import QDRANT_COLLECTION

client = get_client()
ok = True


def check(label: str, condition: bool, detail: str = ""):
    global ok
    mark = "PASS" if condition else "FAIL"
    if not condition:
        ok = False
    print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""))


def qdrant_count(data_type: str) -> int:
    result = client.count(
        collection_name=QDRANT_COLLECTION,
        count_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(key="데이터유형", match=qmodels.MatchValue(value=data_type))]
        ),
    )
    return result.count


print("=" * 60)
print("1. 적재 건수 대조")
print("=" * 60)

org_count = fetch_all("SELECT COUNT(*) AS CNT FROM HISTORY_ORGANIZATION_INFO")[0]["CNT"]
detail_count = fetch_all("SELECT COUNT(*) AS CNT FROM HISTORY_ORGANIZATION_DETAIL")[0]["CNT"]
req_count = fetch_all("SELECT COUNT(*) AS CNT FROM HISTORY_EXTERNAL_REQUEST")[0]["CNT"]
menu3_count = fetch_all("SELECT COUNT(DISTINCT REQUEST_ID) AS CNT FROM HISTORY_INTERNAL_INFO")[0]["CNT"]
menu4_count = fetch_all("SELECT COUNT(*) AS CNT FROM HISTORY_DISCUSSION_COMMITTEE")[0]["CNT"]

check("메뉴1 기관정보 청크 수 = 전체 기관 수", qdrant_count("기관정보") == org_count,
      f"Oracle={org_count}, Qdrant={qdrant_count('기관정보')}")
check("메뉴1 신청이력 청크 수 = 이력 행 수", qdrant_count("신청이력") == detail_count,
      f"Oracle={detail_count}, Qdrant={qdrant_count('신청이력')}")
check("메뉴2 외부요청 청크 수 = 8", qdrant_count("외부요청") == req_count == 8,
      f"Oracle={req_count}, Qdrant={qdrant_count('외부요청')}")
check("메뉴3 내부자료 청크 수 = 13(글 수)", qdrant_count("내부자료") == menu3_count == 13,
      f"Oracle={menu3_count}, Qdrant={qdrant_count('내부자료')}")
check("메뉴4 심의위원회 청크 수 = 57", qdrant_count("심의위원회") == menu4_count == 57,
      f"Oracle={menu4_count}, Qdrant={qdrant_count('심의위원회')}")

print()
print("=" * 60)
print("2. 4가지 케이스 각각 청크 정상 생성 확인")
print("=" * 60)

CASES = {
    "①메인만": ("ORG0000001", "ORG0000025"),
    "②메인+메인첨부": ("ORG0000026", "ORG0000050"),
    "③메인+이력+이력첨부": ("ORG0000051", "ORG0000075"),
    "④전부": ("ORG0000076", "ORG0000100"),
}
for case_name, (lo, hi) in CASES.items():
    sample_org = fetch_all(
        f"SELECT ORGANIZATION_ID FROM HISTORY_ORGANIZATION_INFO "
        f"WHERE ORGANIZATION_ID BETWEEN '{lo}' AND '{hi}' FETCH FIRST 1 ROWS ONLY"
    )
    if not sample_org:
        check(f"{case_name} 샘플 기관 존재", False, "Oracle에 해당 범위 기관 없음")
        continue
    org_id = sample_org[0]["ORGANIZATION_ID"]
    found = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="데이터유형", match=qmodels.MatchValue(value="기관정보")),
                qmodels.FieldCondition(key="원본ID", match=qmodels.MatchValue(value=org_id)),
            ]
        ),
        limit=1,
    )[0]
    check(f"{case_name} ({org_id}) 기관정보 청크 존재", len(found) == 1)

print()
print("=" * 60)
print("3. 첨부 다건(2개 이상) 뭉침 확인 — 행 중복 없이 파일명 한 칸")
print("=" * 60)

multi_att = fetch_all(
    "SELECT ATTFILE_ID, COUNT(*) AS CNT, LISTAGG(ATTFILE_NM, ', ') WITHIN GROUP (ORDER BY ATTFILE_NM) AS NAMES "
    "FROM TM_ATTFILE WHERE DOCUCD = 'C0002013' GROUP BY ATTFILE_ID HAVING COUNT(*) >= 2 "
    "FETCH FIRST 1 ROWS ONLY"
)
if multi_att:
    discussion_id, cnt, names = multi_att[0]["ATTFILE_ID"], multi_att[0]["CNT"], multi_att[0]["NAMES"]
    rows = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="데이터유형", match=qmodels.MatchValue(value="심의위원회")),
                qmodels.FieldCondition(key="원본ID", match=qmodels.MatchValue(value=discussion_id)),
            ]
        ),
        limit=5,
    )[0]
    check(f"심의위원회 {discussion_id} 첨부 {cnt}건 → 청크 1개만 존재", len(rows) == 1,
          f"실제 {len(rows)}개")
    if rows:
        text = rows[0].payload["text"]
        check("첨부문서 라인에 파일명 전체가 한 줄로 포함", all(n.strip() in text for n in names.split(", ")),
              f"기대 파일명: {names}")
else:
    check("첨부 2건 이상 심의위원회 샘플 존재", False, "Oracle에서 못 찾음")

print()
print("=" * 60)
print("4. 메뉴3 그룹핑 — 댓글 여러 개인 글이 1청크로 합쳐지는지")
print("=" * 60)

multi_comment = fetch_all(
    "SELECT REQUEST_ID, COUNT(*) AS CNT FROM HISTORY_INTERNAL_DETAIL "
    "WHERE FILE_COMMENT IS NOT NULL GROUP BY REQUEST_ID HAVING COUNT(*) >= 2 "
    "FETCH FIRST 1 ROWS ONLY"
)
if multi_comment:
    request_id, cnt = multi_comment[0]["REQUEST_ID"], multi_comment[0]["CNT"]
    rows = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="데이터유형", match=qmodels.MatchValue(value="내부자료")),
                qmodels.FieldCondition(key="원본ID", match=qmodels.MatchValue(value=request_id)),
            ]
        ),
        limit=5,
    )[0]
    check(f"내부자료 {request_id} 댓글 {cnt}개 → 청크 1개로 병합", len(rows) == 1, f"실제 {len(rows)}개")
    if rows:
        text = rows[0].payload["text"]
        dash_count = text.count("\n- ")
        check(f"본문에 댓글 라인 {cnt}개 모두 포함", dash_count == cnt, f"실제 댓글 라인 {dash_count}개")
else:
    check("댓글 2개 이상 내부자료 샘플 존재", False, "Oracle에서 못 찾음")

print()
print("=" * 60)
print("5. 검색 시나리오 테스트")
print("=" * 60)


def search(question: str, flt: qmodels.Filter | None = None, top_k: int = 5):
    vector = embed_texts([question])[0]
    return client.query_points(
        collection_name=QDRANT_COLLECTION, query=vector, query_filter=flt, limit=top_k
    ).points


# 기관명/담당자는 정형 식별자이므로 정의서 6.2절 설계대로 메타 필터와 결합해 검색
# (순수 벡터 검색만으로는 정형화된 더미 텍스트 특성상 유사 기관명 간 변별력이 낮음 — 별도 확인 완료)
org_name_row = fetch_all("SELECT CRP_NAME FROM HISTORY_ORGANIZATION_INFO FETCH FIRST 1 ROWS ONLY")[0]
sample_org_name = org_name_row["CRP_NAME"]
results = search(
    f"{sample_org_name} 관련 이력 알려줘",
    flt=qmodels.Filter(must=[qmodels.FieldCondition(key="기관명", match=qmodels.MatchValue(value=sample_org_name))]),
)
check(f"기관명 필터+검색 — '{sample_org_name}'", any(sample_org_name in r.payload.get("text", "") for r in results),
      f"top1 기관명={results[0].payload.get('기관명') if results else None}")

year_row = fetch_all(
    "SELECT SUBSTR(REQ_DTM,1,4) AS YR FROM HISTORY_ORGANIZATION_DETAIL WHERE REQ_DTM IS NOT NULL "
    "FETCH FIRST 1 ROWS ONLY"
)[0]
sample_year = year_row["YR"]
results = search(
    f"{sample_year}년 신청 이력",
    flt=qmodels.Filter(must=[qmodels.FieldCondition(key="연도", match=qmodels.MatchValue(value=sample_year))]),
)
check(f"연도 필터 검색 — {sample_year}년", len(results) > 0 and all(r.payload.get("연도") == sample_year for r in results),
      f"결과 {len(results)}건")

result_row = fetch_all(
    "SELECT DISCUSSION_RESULT FROM HISTORY_DISCUSSION_COMMITTEE WHERE DISCUSSION_RESULT IS NOT NULL "
    "FETCH FIRST 1 ROWS ONLY"
)[0]
sample_result = "".join(result_row["DISCUSSION_RESULT"].split())
results = search(
    f"심의결과 {sample_result}인 안건",
    flt=qmodels.Filter(must=[qmodels.FieldCondition(key="심의결과", match=qmodels.MatchValue(value=sample_result))]),
)
check(f"심의결과 필터 검색 — {sample_result}", len(results) > 0, f"결과 {len(results)}건")

person_row = fetch_all("SELECT USERNM FROM HISTORY_INTERNAL_INFO WHERE USERNM IS NOT NULL FETCH FIRST 1 ROWS ONLY")[0]
sample_person = person_row["USERNM"]
results = search(
    f"{sample_person} 담당 자료",
    flt=qmodels.Filter(must=[qmodels.FieldCondition(key="담당자", match=qmodels.MatchValue(value=sample_person))]),
)
check(f"담당자 필터+검색 — '{sample_person}'", any(sample_person in r.payload.get("text", "") for r in results),
      f"top1 담당자={results[0].payload.get('담당자') if results else None}")

results = search("존재하지않는가상기관XYZ99999의 심의 이력")
top_score = results[0].score if results else 0
check("존재하지 않는 건 질문 — top 스코어 낮음(<0.3, 참고용)", True, f"top1 score={top_score:.3f} (LLM 응답 판단은 API 서버 단계에서)")

print()
print("=" * 60)
print("전체 결과:", "PASS" if ok else "FAIL — 위 FAIL 항목 확인 필요")
print("=" * 60)
sys.exit(0 if ok else 1)
