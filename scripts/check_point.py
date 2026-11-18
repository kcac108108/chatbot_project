"""특정 데이터(기관/이력/안건 등) 하나가 실제로 벡터DB(Qdrant)에 임베딩까지 끝나고
들어가 있는지 직접 확인하는 진단 도구(2026-11-06).

증분 적재 배치의 CHUNK_COUNT 같은 숫자는 "몇 건 처리했다"는 집계일 뿐, "내가 등록한
그 특정 이력이 실제로 들어갔는지"를 보려면 Oracle 여러 테이블을 역산해야 했다 — 이
스크립트는 그 역산 없이, (데이터유형, 원본ID) 하나만 알면 Qdrant에서 직접 찾아서
바로 확인해준다.

사용법:
    python scripts/check_point.py <데이터유형> <원본ID>

예시:
    python scripts/check_point.py 신청이력 D20260100179700
    python scripts/check_point.py 기관정보 ORG0012

데이터유형: 기관정보 / 신청이력 / 외부요청 / 내부자료 / 심의위원회
          (첨부는 각 메뉴명 + "첨부", 예: 신청이력첨부 / 심의위원회첨부)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.vectordb import find_by_type_and_source

# Windows 콘솔(cp949)에서 이모지·일부 유니코드 출력 시 UnicodeEncodeError가 나는 걸
# 방지 — 운영(리눅스 컨테이너)은 기본이 UTF-8이라 원래도 문제없지만, 이 PC에서
# 바로 돌려볼 때도 안전하게 동작하도록 명시적으로 맞춰준다.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    data_type, source_id = sys.argv[1], sys.argv[2]
    payloads = find_by_type_and_source(data_type, source_id)

    if not payloads:
        print(f"[확인 결과] ❌ 없음 — 데이터유형='{data_type}', 원본ID='{source_id}'로 "
              f"Qdrant에 저장된 청크가 하나도 없습니다.")
        print("          아직 배치가 안 돌았거나, 데이터유형/원본ID 값이 정확한지 다시 확인해보세요.")
        return

    print(f"[확인 결과] ✅ 있음 — {len(payloads)}개 청크 확인됨 "
          f"(데이터유형='{data_type}', 원본ID='{source_id}')\n")

    for i, p in enumerate(payloads, 1):
        print(f"--- 청크 {i}/{len(payloads)} ---")
        for key in ("기관명", "연도", "통계종류", "요청자", "담당자", "심의결과", "첨부파일명"):
            if p.get(key):
                print(f"  {key}: {p[key]}")
        if p.get("세부항목"):
            print("  세부항목:")
            for item in p["세부항목"][:5]:
                print(f"    - {item.get('label')}: {item.get('value')}")
        text = (p.get("text") or "")[:200]
        print(f"  본문 앞부분: {text}{'...' if len(p.get('text') or '') > 200 else ''}")
        print()


if __name__ == "__main__":
    main()
