"""신규 이력 첨부파일 자동처리 파이프라인 — 마지막 실행 이후 새로 등록되거나 수정된
첨부파일만 찾아, 청킹·요약·임베딩까지 자동으로 반영한다.

**OCR/텍스트 추출은 전산팀이 담당한다(2026-10-12 확정)**: 전산팀이 매일 새벽 1시부터
그날 신규 생성된 첨부파일을 자체 배치로 처리해서, 원본과 같은 경로·같은 파일명(확장자만
`.txt`)으로 결과 텍스트 파일을 만들어준다 — 스캔본(OCR 필요)·엑셀·한글·zip(내부 파일들을
합쳐 결과 1개) 등 형식 무관하게 전부 이 규칙 하나로 처리됨. 그래서 우리 쪽은 그 결과
파일을 "읽기만" 하면 되고, 우리가 직접 OCR API를 호출하던 이전 설계(`ocr_remote.py`,
2026-09~10/1 작업분)와 엑셀/한글 직접추출(`attachment_extract.extract_any`)은 이 자동
파이프라인에서는 더 이상 쓰지 않는다(수동 백필/재검토 용도로는 여전히 유효하므로 파일
자체는 남겨둠).

기존 수동 파이프라인(`attachment_ingest.ingest_attachments`)과 공유하는 부분: DOCUCD별
매핑, 경로전용(카탈로그성) 처리, 청킹·요약·임베딩 로직(`_build_entries_for_text`/
`_upsert_entries`).

**전산팀 배치 타이밍**: 새벽 1시 시작, 종료 시각은 고정이 아니지만 넉넉히 오전 9시에
우리 쪽을 실행하면 안전하다고 판단(사용자 확인, 2026-10-12).

**결과 파일이 없을 수 있음**: 배치가 특정 파일에서 실패하거나, 자정 무렵 생성된 첨부가
"전일자" 기준 경계에 걸려 그날 배치에 안 잡혔을 수 있다 — 조용히 건너뛰고 다음 실행에서도
놓치지 않도록, 그런 건이 하나라도 있으면 이번 실행의 "마지막 처리 시각" 워터마크를
그 건 바로 이전으로 고정해서 다음 실행에 다시 조회되게 한다(아래 `_record_attachment_ingest_run`
호출부 참고). 이미 성공한 건도 같이 재처리될 수 있지만, Qdrant point ID가 결정론적이라
그냥 덮어써질 뿐 중복 적재되지는 않는다.
"""
from datetime import datetime, timedelta

from app.core.db import db_now, execute, fetch_all
from app.services.attachment_ingest import (
    _DOCUCD_MAP,
    _build_entries_for_text,
    _is_path_only,
    _result_text_path,
    _upsert_entries,
    _valid_attachments,
)


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


def _attachment_effective_dtm(att: dict) -> datetime:
    """_valid_attachments의 since 필터와 같은 기준 — CR_DTM/IN_DTM 중 더 최근 값."""
    candidates = [dt for dt in (att.get("CR_DTM"), att.get("IN_DTM")) if dt]
    return max(candidates) if candidates else datetime.min


def run_incremental_attachment_ingest(docucd: str) -> int:
    """DOCUCD 하나에 대해 마지막 실행 이후 신규·수정된 첨부만 골라 처리한다.
    카테고리별로 마지막 실행 시각을 따로 추적하므로(CHATBOT_ATTACHMENT_INGEST_LOG),
    메뉴마다 원하는 주기로 따로 돌려도 된다."""
    if docucd not in _DOCUCD_MAP:
        raise ValueError(f"알 수 없는 DOCUCD: {docucd}")
    _, _, data_type = _DOCUCD_MAP[docucd]

    run_started_at = db_now()
    since = _get_last_attachment_ingest_at(docucd)
    new_atts = _valid_attachments(docucd, since=since)
    print(f"[attachment_auto:{docucd}] 신규·수정 첨부 {len(new_atts)}건 발견 (기준시각: {since})")

    if not new_atts:
        _record_attachment_ingest_run(docucd, run_started_at, 0)
        return 0

    entries: list[tuple[str, dict]] = []
    pending_dtms: list[datetime] = []  # 결과파일이 아직 없는 건들의 등록·수정시각

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

        result_path = _result_text_path(att["FILE_PATH"], docucd, parent_id)
        if not result_path.exists():
            print(f"[attachment_auto:{docucd}] 결과파일 없음(보류, 다음 실행에 재시도): {filename}")
            pending_dtms.append(_attachment_effective_dtm(att))
            continue

        # 결과 파일 인코딩은 UTF-8로 전산팀에 확인됨(2026-10-12) — errors="replace"로 깨진
        # 글자를 조용히 대체문자로 덮지 않는다. 예상과 달리 UTF-8이 아닌 파일이 섞여 있으면
        # 조용히 깨진 데이터를 적재하는 것보다, 여기서 예외로 드러나 그 건이 눈에 띄는 게
        # 낫다(건별 try/except 없이 run_incremental_attachment_ingest_all()의 메뉴 단위
        # catch-all로 올라가 그 메뉴 전체가 "오류로 중단"되는 점은 감수 — 과거 SQL 파일
        # CP949 오독 버그처럼 "조용히 깨지는 것"이 가장 위험하다는 교훈 반영).
        text = result_path.read_text(encoding="utf-8")
        if not text.strip():
            print(f"[attachment_auto:{docucd}] 결과파일 비어있음(건너뜀): {filename}")
            continue
        print(f"[attachment_auto:{docucd}] 결과파일 읽음({len(text)}자): {filename}")
        entries += _build_entries_for_text(parent_id, att["ATTFILE_NUM"], filename, text, data_type)

    total = _upsert_entries(data_type, entries, f"attachment_auto:{docucd}") if entries else 0

    if pending_dtms:
        # 가장 오래된 보류 건 바로 이전까지만 워터마크를 전진시켜, 다음 실행에서 그 건이
        # (그리고 그 이후에 생긴 다른 건들도 같이) 다시 조회 대상에 걸리게 한다.
        next_watermark = min(pending_dtms) - timedelta(seconds=1)
        print(f"[attachment_auto:{docucd}] 보류 {len(pending_dtms)}건 — 다음 실행 기준시각을 {next_watermark}로 고정")
    else:
        next_watermark = run_started_at
    _record_attachment_ingest_run(docucd, next_watermark, total)
    return total


def run_incremental_attachment_ingest_all() -> dict[str, int]:
    """5개 메뉴(DOCUCD) 전부 순서대로 증분 처리. 한 메뉴가 실패해도(예: 파일없음 다수)
    나머지 메뉴는 계속 진행하고, 실패한 메뉴는 결과에 -1로 표시한다."""
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
