"""신규 이력 첨부파일 자동처리 파이프라인(2026-09-29) — 마지막 실행 이후 새로
등록되거나 수정된 첨부파일만 찾아, 필요하면 전산팀 OCR 서버로 문자를 추출한 뒤
청킹·요약·임베딩까지 자동으로 반영한다.

기존 수동 파이프라인(`attachment_ingest.ingest_attachments`)과의 관계:
- DOCUCD별 매핑, 경로전용(카탈로그성) 처리, 청킹·요약·임베딩 로직은 전부 그대로 재사용한다
  (`_build_entries_for_text`/`_upsert_entries`).
- 직접추출 가능한 형식(엑셀·한글·텍스트 등)과 zip은 기존 `extract_any`를 그대로 쓴다
  (zip 내부에 스캔본이 섞여 있으면 기존처럼 OpenAI 비전모델로 처리 — zip은 물량이 적어
  전산팀 OCR 연동까지는 아직 확장하지 않음, 필요해지면 나중에 확장).
- **스캔본 PDF·이미지만** 기존 OpenAI 비전모델 대신 전산팀 OCR 서버(`ocr_remote`)로
  라우팅한다 — 내부 OCR 방식이 이미 품질·비용 면에서 채택 확정된 방향이라(2026-09 초
  대량 적재 때 결정) 신규 건도 같은 방식으로 통일.

⚠️ SFTP 접속정보(포트·계정)가 전산팀 호환성 조사 완료 전까지는 비어있다 — 그 상태에서
실행하면 `ocr_remote.OCRConfigError`로 명확하게 실패한다(경우 따라 실행 자체를 미리
건너뛰고 안내만 하고 싶으면 `check_ocr_config()`를 먼저 불러 확인할 것).
"""
from datetime import datetime
from pathlib import Path

from app.core.config import ATTACHMENT_BASE_DIR
from app.core.db import execute, fetch_all
from app.services.attachment_extract import extract_any
from app.services.attachment_ingest import (
    _DOCUCD_MAP,
    _build_entries_for_text,
    _is_path_only,
    _local_path,
    _upsert_entries,
    _valid_attachments,
)
from app.services.ocr_remote import check_ocr_config, run_ocr_batch

_OCR_EXTS = (".pdf", ".png", ".jpg", ".jpeg")


def _get_last_attachment_ingest_at(docucd: str) -> datetime | None:
    rows = fetch_all(
        "SELECT MAX(RUN_AT) AS LAST_RUN_AT FROM CHATBOT_ATTACHMENT_INGEST_LOG WHERE DOCUCD = :docucd",
        {"docucd": docucd},
    )
    return rows[0]["LAST_RUN_AT"] if rows else None


def _record_attachment_ingest_run(docucd: str, run_at: datetime, chunk_count: int) -> None:
    execute(
        "INSERT INTO CHATBOT_ATTACHMENT_INGEST_LOG (DOCUCD, RUN_AT, CHUNK_COUNT) "
        "VALUES (:docucd, :run_at, :chunk_count)",
        {"docucd": docucd, "run_at": run_at, "chunk_count": chunk_count},
    )


def run_incremental_attachment_ingest(docucd: str) -> int:
    """DOCUCD 하나에 대해 마지막 실행 이후 신규·수정된 첨부만 골라 처리한다.
    카테고리별로 마지막 실행 시각을 따로 추적하므로(CHATBOT_ATTACHMENT_INGEST_LOG),
    메뉴마다 원하는 주기로 따로 돌려도 된다.

    ⚠️ 이 함수 자체(및 내부에서 호출하는 ocr_remote.run_ocr_batch)는 동시에 두 곳에서
    실행하면 안 된다 — 전산팀 확인상 OCR 서버 쪽 처리가 겹치면 파일이 덮어써진다.
    스케줄러에 등록할 때 겹쳐 실행되지 않도록(예: 이전 실행이 안 끝났으면 건너뛰기)
    호출부에서 보장해야 한다."""
    if docucd not in _DOCUCD_MAP:
        raise ValueError(f"알 수 없는 DOCUCD: {docucd}")
    _, _, data_type = _DOCUCD_MAP[docucd]

    run_started_at = datetime.now()
    since = _get_last_attachment_ingest_at(docucd)
    new_atts = _valid_attachments(docucd, since=since)
    print(f"[attachment_auto:{docucd}] 신규·수정 첨부 {len(new_atts)}건 발견 (기준시각: {since})")

    if not new_atts:
        _record_attachment_ingest_run(docucd, run_started_at, 0)
        return 0

    entries: list[tuple[str, dict]] = []
    ocr_targets: list[tuple[dict, Path]] = []  # (att, local_path) — OCR 서버로 보낼 것들

    for att in new_atts:
        filename = att["ATTFILE_NM"]
        parent_id = att["ATTFILE_ID"]

        if _is_path_only(filename):
            entries.append((
                f"{parent_id}:{att['ATTFILE_NUM']}:0",
                {
                    "text": f"[{data_type} {parent_id} / {filename}] (테이블/SQL 정보 파일 - 내용 미제공)",
                    "데이터유형": data_type,
                    "원본ID": parent_id,
                    "첨부파일명": filename,
                    "경로전용": True,
                    "파일경로": att["FILE_PATH"],
                },
            ))
            print(f"[attachment_auto:{docucd}] 경로만 표시: {filename}")
            continue

        path = _local_path(att["FILE_PATH"], docucd, parent_id)
        if not path.exists():
            print(f"[attachment_auto:{docucd}] 파일없음(건너뜀): {filename}")
            continue

        if path.suffix.lower() in _OCR_EXTS:
            ocr_targets.append((att, path))
            continue

        # 직접추출 가능한 형식(엑셀·한글·텍스트 등)이나 zip — 기존 extract_any 그대로 사용.
        text = extract_any(path)
        if not text or not text.strip():
            print(f"[attachment_auto:{docucd}] 추출결과없음(건너뜀): {filename}")
            continue
        print(f"[attachment_auto:{docucd}] 직접추출 완료({len(text)}자): {filename}")
        entries += _build_entries_for_text(parent_id, att["ATTFILE_NUM"], filename, text, data_type)

    if ocr_targets:
        check_ocr_config()
        print(f"[attachment_auto:{docucd}] OCR 대상 {len(ocr_targets)}건 — 전산팀 서버로 전송")
        succeeded, failed = run_ocr_batch([p for _, p in ocr_targets])
        for att, path in ocr_targets:
            filename = att["ATTFILE_NM"]
            if path in failed:
                print(f"[attachment_auto:{docucd}] OCR 실패(건너뜀): {filename} — {failed[path]}")
                continue
            text = succeeded.get(path)
            if not text or not text.strip():
                print(f"[attachment_auto:{docucd}] OCR 결과 없음(건너뜀): {filename}")
                continue
            print(f"[attachment_auto:{docucd}] OCR 완료({len(text)}자): {filename}")
            entries += _build_entries_for_text(
                att["ATTFILE_ID"], att["ATTFILE_NUM"], filename, text, data_type
            )

    total = _upsert_entries(data_type, entries, f"attachment_auto:{docucd}")
    _record_attachment_ingest_run(docucd, run_started_at, total)
    return total


def run_incremental_attachment_ingest_all() -> dict[str, int]:
    """5개 메뉴(DOCUCD) 전부 순서대로 증분 처리. 한 메뉴가 실패해도(예: 파일없음 다수,
    OCR 설정 미비) 나머지 메뉴는 계속 진행하고, 실패한 메뉴는 결과에 -1로 표시한다."""
    results: dict[str, int] = {}
    for docucd in _DOCUCD_MAP:
        try:
            results[docucd] = run_incremental_attachment_ingest(docucd)
        except Exception as e:
            print(f"[attachment_auto:{docucd}] 실행 중 오류로 중단: {e}")
            results[docucd] = -1
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in _DOCUCD_MAP:
        count = run_incremental_attachment_ingest(sys.argv[1])
        print(f"[attachment_auto] {sys.argv[1]} 완료: {count}개 청크 적재")
    else:
        results = run_incremental_attachment_ingest_all()
        print(f"[attachment_auto] 전체 완료: {results}")
