"""첨부파일 전체 다운로드 지원 확장(2026-11-05) 백필 스크립트.

그 전까지는 경로전용(SQL/카탈로그성) 첨부에만 Qdrant payload에 "파일경로"가
저장돼 있었다. 이제부터는 일반 첨부도 적재 시점에 파일경로를 같이 저장하도록
코드를 바꿨지만(app/services/attachment_ingest.py의 _build_entries_for_text),
이미 적재돼 있는 기존 청크에는 이 필드가 없다.

TM_ATTFILE에서 (ATTFILE_ID, ATTFILE_NM) -> FILE_PATH 매핑을 만들어, 그 필드가
없는 기존 청크에 Qdrant set_payload로만 채운다 — 재임베딩 불필요(파일경로는
순수 메타데이터라 임베딩 대상 text에 포함되지 않음, 2026-09-18에 이미 같은
방식으로 "요약" 필드를 백필했던 전례와 동일 패턴).

사용법: python scripts/backfill_attachment_file_path.py [--dry-run]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import QDRANT_COLLECTION
from app.core.vectordb import get_client
from app.services.attachment_ingest import _DOCUCD_MAP, _valid_attachments
from qdrant_client.http import models as qmodels


def build_lookup() -> dict[tuple[str, str], str]:
    """(원본ID, 첨부파일명) -> TM_ATTFILE.FILE_PATH 매핑. 5개 메뉴 전부 순회."""
    lookup: dict[tuple[str, str], str] = {}
    for docucd in _DOCUCD_MAP:
        atts = _valid_attachments(docucd)
        for att in atts:
            lookup[(att["ATTFILE_ID"], att["ATTFILE_NM"])] = att["FILE_PATH"]
        print(f"[{docucd}] TM_ATTFILE {len(atts)}건 조회")
    return lookup


def run(dry_run: bool) -> None:
    lookup = build_lookup()
    print(f"전체 매핑 {len(lookup)}건 구성 완료")

    client = get_client()
    # _DOCUCD_MAP의 세 번째 값은 attachment_ingest.py가 payload에 그대로 쓰는 데이터유형
    # 문자열이라, 이미 "기관정보첨부"처럼 첨부 타입 자체다(부모 타입이 아님) — 그대로 쓴다.
    attachment_types = {m[2] for m in _DOCUCD_MAP.values()}
    print(f"대상 데이터유형: {sorted(attachment_types)}")

    updated = 0
    missing = 0
    already_had = 0
    skipped_path_only = 0
    offset = None
    scanned = 0
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = p.payload
            if payload.get("데이터유형") not in attachment_types:
                continue
            scanned += 1
            if payload.get("경로전용"):
                skipped_path_only += 1
                continue
            if payload.get("파일경로"):
                already_had += 1
                continue
            key = (payload.get("원본ID"), payload.get("첨부파일명"))
            file_path = lookup.get(key)
            if not file_path:
                missing += 1
                print(f"  매칭 실패: {key}")
                continue
            if not dry_run:
                client.set_payload(
                    collection_name=QDRANT_COLLECTION,
                    payload={"파일경로": file_path},
                    points=[p.id],
                )
            updated += 1
        if offset is None:
            break

    print(
        f"\n스캔한 첨부 청크 {scanned}건 중: 신규 반영 {updated}건, "
        f"이미 있음 {already_had}건, 경로전용(대상 아님) {skipped_path_only}건, "
        f"매칭 실패 {missing}건"
    )
    if dry_run:
        print("(--dry-run: 실제 Qdrant 반영 없이 집계만 수행)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.dry_run)
