"""배포 당일 1회용 — 로컬DB 마이그레이션 이후 운영 DB에 쌓인 데이터를 수동으로 캐치업 적재한다.

운영 서버의 CHATBOT_INGEST_LOG/CHATBOT_ATTACHMENT_INGEST_LOG는 배포 당일 새로 생성되어
비어있는 상태다. 이 상태로 run_incremental_ingest()를 그냥 돌리면 "기준 시점 없음"으로
보고 운영 DB 전체를 처음부터 다시 처리한다(안전하지만 느리고 비용이 큼 — 특히 첨부파일
~2,000여개 전부 재요약·재임베딩).

이 스크립트는 로컬DB 마이그레이션이 이루어진 기준일을 직접 지정해 그 날짜로 두 로그
테이블에 워터마크를 먼저 심어두고, 그다음 기존에 이미 검증된 증분 적재 함수
(run_incremental_ingest / run_incremental_attachment_ingest_all)를 그대로 호출한다 —
새 로직을 만들지 않고 기존 코드를 그대로 재사용하므로 안전하다.

기준일은 실제 마이그레이션 날짜보다 다소 여유있게(며칠 이르게) 잡는 것을 권장한다.
Qdrant point ID가 (데이터유형, 원본ID) 기준 결정론적이라, 이미 적재된 레코드가 겹쳐서
다시 처리되어도 중복 생성 없이 덮어쓰기(upsert)만 되므로 "너무 이르게" 잡아서 생기는
위험은 없다 — 반대로 "너무 늦게" 잡으면 그 사이 데이터를 놓칠 수 있으니 반드시 안전한
쪽(이르게)으로 잡을 것.

실행(운영서버에서, .env가 운영 설정 — DB_TABLE_SUFFIX=@STATWEB, QDRANT_HOST=qdrant 등
이미 적용된 상태):
    python scripts/manual_catchup_ingest.py --since 2026-08-18

실행 후에는 cron으로 등록된 정기 증분 적재가 이 시점부터 정상적으로 이어진다.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.db import execute
from app.services.attachment_ingest import _DOCUCD_MAP
from app.services.attachment_auto_ingest import run_incremental_attachment_ingest_all
from app.services.ingest import run_incremental_ingest


def _seed_watermarks(since: datetime) -> None:
    execute(
        "INSERT INTO CHATBOT_INGEST_LOG (RUN_AT, CHUNK_COUNT) VALUES (:run_at, :chunk_count)",
        {"run_at": since, "chunk_count": 0},
    )
    for docucd in _DOCUCD_MAP:
        execute(
            "INSERT INTO CHATBOT_ATTACHMENT_INGEST_LOG (DOCUCD, RUN_AT, CHUNK_COUNT) "
            "VALUES (:docucd, :run_at, :chunk_count)",
            {"docucd": docucd, "run_at": since, "chunk_count": 0},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="마이그레이션 기준일 이후 운영 DB 캐치업 적재(1회용)")
    parser.add_argument(
        "--since",
        required=True,
        help="마이그레이션 기준일 (YYYY-MM-DD). 안전하게 실제보다 며칠 이르게 잡을 것.",
    )
    args = parser.parse_args()
    since = datetime.strptime(args.since, "%Y-%m-%d")

    print(f"=== 기준시각 {since} 로 워터마크 등록 ===")
    _seed_watermarks(since)

    print("=== 메뉴1~4(정형 데이터) 캐치업 적재 시작 ===")
    count = run_incremental_ingest()
    print(f"메뉴1~4: {count}개 청크 적재 완료")

    print("=== 첨부파일 캐치업 적재 시작 ===")
    results = run_incremental_attachment_ingest_all()
    print(f"첨부파일: {results}")

    print("=== 완료 — 이후 cron 증분 적재가 정상적으로 이어집니다 ===")


if __name__ == "__main__":
    main()
