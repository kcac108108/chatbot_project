"""운영 Oracle에서 export한 Excel 파일을 로컬 Oracle에 UPSERT한다 (수동 데이터 동기화용).

2026-10-27 "8/18 이후 누락 데이터 캐치업" 때와 동일한 패턴 — 운영 DB에서 마지막 동기화
시점 이후 바뀐/추가된 행만 SQL로 뽑아 Excel로 export 받은 뒤, 겹치는 행은 UPDATE,
없으면 INSERT(MERGE INTO)로 로컬 Oracle에 반영한다.

전제:
- 각 Excel 파일명이 테이블명과 정확히 같아야 한다(예: HISTORY_ORGANIZATION_INFO.xlsx).
- 컬럼명이 Oracle 컬럼명과 정확히 일치해야 한다(SELECT * 로 뽑은 그대로).
- DATE 컬럼 값은 'YYYY-MM-DD HH24:MI:SS' 형식의 문자열(pandas가 읽은 그대로)이어야 한다.

실행: venv/Scripts/python.exe scripts/sync_oracle_from_excel.py "<엑셀 파일들이 있는 폴더>"
"""
import sys
from datetime import datetime
from math import isnan
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.core.db import execute, fetch_all

# 테이블별: (PK 컬럼 목록, 전체 컬럼 목록, DATE 컬럼 목록)
_TABLES: dict[str, dict] = {
    "HISTORY_ORGANIZATION_INFO": dict(
        pk=["ORGANIZATION_ID"],
        columns=[
            "ORGANIZATION_ID", "CRP_NO", "CRP_NAME", "CRP_DEPT_NAME", "BUSINESS_NAME",
            "LAW_COMMENT", "SUPPORT_COMMENT", "CRP_GBN", "CONTENT",
            "IN_DTM", "IN_USERID", "CR_DTM", "CR_USERID",
        ],
        dates=["IN_DTM", "CR_DTM"],
    ),
    "HISTORY_ORGANIZATION_DETAIL": dict(
        pk=["DETAIL_ID"],
        columns=[
            "ORGANIZATION_ID", "DETAIL_ID", "REQ_DTM", "TRNS_MTD", "STS_ISSUE_TYPE",
            "STS_ISSUE_GBN", "STS_ISSUE_ITEM", "TRNS_CYCLE", "REQ_NOTE", "REQ_GBN",
            "IN_DTM", "IN_USERID", "CR_DTM", "CR_USERID", "PAY_AMT",
        ],
        dates=["IN_DTM", "CR_DTM"],
    ),
    "HISTORY_EXTERNAL_REQUEST": dict(
        pk=["REQUEST_ID"],
        columns=[
            "REQUEST_ID", "CRP_NO", "CRP_NAME", "CRP_DEPT_NAME", "REQ_PERSON_NAME",
            "REQ_DTM", "REQ_TEL", "PROC_PERSON_NAME", "TRNS_MTD", "TITLE", "REQ_MENT",
            "PROC_DTM", "PROC_MENT", "NOTE", "IN_DTM", "IN_USERID", "CR_DTM", "CR_USERID",
        ],
        dates=["IN_DTM", "CR_DTM"],
    ),
    "HISTORY_DISCUSSION_COMMITTEE": dict(
        pk=["DISCUSSION_ID"],
        columns=[
            "DISCUSSION_ID", "CRP_NO", "CRP_NAME", "CRP_DEPT_NAME", "REQ_PERSON_NAME",
            "REQ_DTM", "REQ_TEL", "TITLE", "REQ_MENT", "LAW_COMMENT", "NOTE",
            "AGENDA_NUMBER", "YEAR", "SEQ", "DISCUSSION_RESULT", "RESULT_CONTENT",
            "REQ_ITEM", "PROC_ITEM", "PROC_DTM", "IN_DTM", "IN_USERID", "CR_DTM", "CR_USERID",
        ],
        dates=["IN_DTM", "CR_DTM"],
        ints=["SEQ"],
    ),
    "TM_ATTFILE": dict(
        pk=["DOCUCD", "ATTFILE_ID", "ATTFILE_NUM"],
        columns=[
            "DOCUCD", "ATTFILE_ID", "ATTFILE_NUM", "ATTFILE_NM", "FILE_PATH", "THUMB_PATH",
            "FILE_EXT", "FILE_SIZE", "DOWNLOAD_CNT", "CR_USERID", "CR_DTM", "IN_USERID", "IN_DTM",
        ],
        dates=["CR_DTM", "IN_DTM"],
        ints=["ATTFILE_NUM", "DOWNLOAD_CNT"],
        floats=["FILE_SIZE"],
    ),
}


def _clean_value(raw, is_date: bool, is_int: bool, is_float: bool):
    if raw is None:
        return None
    if isinstance(raw, float) and isnan(raw):
        return None
    if isinstance(raw, str) and raw.strip() == "":
        return None
    if is_date:
        if isinstance(raw, datetime):
            return raw
        return datetime.strptime(str(raw).strip(), "%Y-%m-%d %H:%M:%S")
    if is_int:
        return int(float(raw))
    if is_float:
        return float(raw)
    return str(raw)


def _merge_sql(table: str, columns: list[str], pk: list[str]) -> str:
    non_pk = [c for c in columns if c not in pk]
    on_clause = " AND ".join(f"t.{c} = s.{c}" for c in pk)
    select_cols = ", ".join(f":{c} AS {c}" for c in columns)
    set_clause = ", ".join(f"t.{c} = s.{c}" for c in non_pk) if non_pk else None
    insert_cols = ", ".join(columns)
    insert_vals = ", ".join(f"s.{c}" for c in columns)
    when_matched = f"WHEN MATCHED THEN UPDATE SET {set_clause}\n" if set_clause else ""
    return (
        f"MERGE INTO {table} t\n"
        f"USING (SELECT {select_cols} FROM DUAL) s\n"
        f"ON ({on_clause})\n"
        f"{when_matched}"
        f"WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})"
    )


def sync_table(table: str, folder: Path) -> tuple[int, int]:
    cfg = _TABLES[table]
    path = folder / f"{table}.xlsx"
    if not path.is_file():
        print(f"  [{table}] 파일 없음 — 건너뜀")
        return 0, 0

    df = pd.read_excel(path, dtype=str)
    columns = cfg["columns"]
    pk = cfg["pk"]
    dates = set(cfg.get("dates", []))
    ints = set(cfg.get("ints", []))
    floats = set(cfg.get("floats", []))

    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[{table}] 엑셀에 없는 컬럼: {missing_cols}")

    sql = _merge_sql(table, columns, pk)

    # 반영 전/후 건수 비교용으로, PK 조건으로 기존 존재 여부를 세어둔다(통계 출력용일 뿐,
    # MERGE 자체의 동작에는 영향 없음).
    inserted = 0
    updated = 0
    for _, row in df.iterrows():
        params = {
            c: _clean_value(row[c], c in dates, c in ints, c in floats)
            for c in columns
        }
        pk_where = " AND ".join(f"{c} = :{c}" for c in pk)
        pk_params = {c: params[c] for c in pk}
        existing = fetch_all(f"SELECT COUNT(*) AS CNT FROM {table} WHERE {pk_where}", pk_params)
        if existing[0]["CNT"] > 0:
            updated += 1
        else:
            inserted += 1
        execute(sql, params)

    print(f"  [{table}] 총 {len(df)}건 처리 — 신규 {inserted}건, 갱신 {updated}건")
    return inserted, updated


def main() -> None:
    if len(sys.argv) != 2:
        print("사용법: python scripts/sync_oracle_from_excel.py <엑셀 파일들이 있는 폴더>")
        sys.exit(1)
    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"폴더를 찾을 수 없음: {folder}")
        sys.exit(1)

    print(f"=== {folder} 에서 동기화 시작 ===")
    total_ins, total_upd = 0, 0
    # 참조 관계상 자연스러운 순서(INFO -> DETAIL -> 나머지)로 처리 — DB에 FK 강제는 없지만
    # 논리적으로 맞춰둔다.
    for table in [
        "HISTORY_ORGANIZATION_INFO",
        "HISTORY_ORGANIZATION_DETAIL",
        "HISTORY_EXTERNAL_REQUEST",
        "HISTORY_DISCUSSION_COMMITTEE",
        "TM_ATTFILE",
    ]:
        ins, upd = sync_table(table, folder)
        total_ins += ins
        total_upd += upd
    print(f"=== 완료 — 전체 신규 {total_ins}건, 갱신 {total_upd}건 ===")


if __name__ == "__main__":
    main()
