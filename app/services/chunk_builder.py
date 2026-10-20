from app.services.cleanse import cleanse


def _line(label: str, value) -> str | None:
    value = cleanse(value) if isinstance(value, str) else value
    if value is None or value == "":
        return None
    return f"{label}: {value}"


def _parts(*pairs) -> str | None:
    items = []
    for label, value in pairs:
        value = cleanse(value) if isinstance(value, str) else value
        if value is not None and value != "":
            items.append(f"{label}: {value}")
    return " / ".join(items) if items else None


def _fmt_date(value) -> str | None:
    if value is None:
        return None
    return value.strftime("%Y-%m-%d")


def _fmt_date_str(value: str | None) -> str | None:
    """REQ_DTM 등 VARCHAR2(8) "YYYYMMDD" 형식 문자열용 — datetime 객체가 아니라 _fmt_date 못 씀."""
    if not value or len(value) != 8:
        return value
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def _detail_items(*pairs) -> list[dict]:
    """근거자료 미리보기용 — 원본 컬럼값을 라벨별로 나눠 저장(값 없으면 제외)."""
    items = []
    for label, value in pairs:
        value = cleanse(value) if isinstance(value, str) else value
        if value is not None and value != "":
            items.append({"label": label, "value": str(value)})
    return items


def _normalize_result(value: str | None) -> str | None:
    if value is None:
        return None
    return "".join(value.split())


# ------------------------------------------------------------------
# 메뉴 1: 기관별 사업정보 — 기관 1건당 [기관정보] 1개 + 이력 1건당 [신청이력] 1개
# ------------------------------------------------------------------
def build_menu1_chunks(rows: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    seen_orgs: set[str] = set()

    for row in rows:
        org_id = row["ORGANIZATION_ID"]

        if org_id not in seen_orgs:
            seen_orgs.add(org_id)
            lines = [
                f"[기관정보] {cleanse(row['CRP_NAME'])}({row['CRP_GBN']}, {cleanse(row['CRP_DEPT_NAME']) or '부서 미상'})",
                _line("사업명", row["BUSINESS_NAME"]),
                _line("법적근거", row["LAW_COMMENT"]),
                _line("제공목적", row["SUPPORT_COMMENT"]),
                _line("주요내용", row["CONTENT"]),
                _line("기관자료", row["ORG_ATT_NAMES"]),
            ]
            chunks.append({
                "text": "\n".join(l for l in lines if l),
                "metadata": {
                    "데이터유형": "기관정보",
                    "원본ID": org_id,
                    "사업자번호": row["CRP_NO"],
                    "기관명": row["CRP_NAME"],
                    "기관분류": row["CRP_GBN"],
                    "세부항목": _detail_items(
                        ("사업명", row["BUSINESS_NAME"]),
                        ("법적근거", row["LAW_COMMENT"]),
                        ("제공목적", row["SUPPORT_COMMENT"]),
                        ("주요내용", row["CONTENT"]),
                        ("기관자료", row["ORG_ATT_NAMES"]),
                    ),
                },
            })

        if row["DETAIL_ID"] is not None:
            lines = [
                f"[신청이력 {row['DETAIL_ID']}] {cleanse(row['CRP_NAME'])}"
                f"({row['CRP_GBN']}, {cleanse(row['CRP_DEPT_NAME']) or '부서 미상'})이(가) {row['REQ_DTM']} 신청.",
                _line("사업명", row["BUSINESS_NAME"]),
                _line("제공목적", row["SUPPORT_COMMENT"]),
                _parts(("요청구분", row["REQ_GBN"]), ("통계종류", row["STS_ISSUE_TYPE"]), ("수출입구분", row["STS_ISSUE_GBN"])),
                _line("제공항목", row["STS_ISSUE_ITEM"]),
                _parts(("전송방식", row["TRNS_MTD"]), ("제공주기", row["TRNS_CYCLE"]), ("수수료", row["PAY_AMT"])),
                _line("요청내용", row["REQ_NOTE"]),
                _line("기관자료", row["ORG_ATT_NAMES"]),
                _line("첨부문서", row["DTL_ATT_NAMES"]),
            ]
            chunks.append({
                "text": "\n".join(l for l in lines if l),
                "metadata": {
                    "데이터유형": "신청이력",
                    "원본ID": row["DETAIL_ID"],
                    "기관ID": org_id,
                    "사업자번호": row["CRP_NO"],
                    "기관명": row["CRP_NAME"],
                    "기관분류": row["CRP_GBN"],
                    "연도": row["REQ_DTM"][:4] if row["REQ_DTM"] else None,
                    "통계종류": row["STS_ISSUE_TYPE"],
                    "수출입구분": row["STS_ISSUE_GBN"],
                    "세부항목": _detail_items(
                        ("사업명", row["BUSINESS_NAME"]),
                        ("신청일", _fmt_date_str(row["REQ_DTM"])),
                        ("요청구분", row["REQ_GBN"]),
                        ("통계종류", row["STS_ISSUE_TYPE"]),
                        ("수출입구분", row["STS_ISSUE_GBN"]),
                        ("제공항목", row["STS_ISSUE_ITEM"]),
                        ("전송방식", row["TRNS_MTD"]),
                        ("제공주기", row["TRNS_CYCLE"]),
                        ("수수료", row["PAY_AMT"]),
                        ("요청내용", row["REQ_NOTE"]),
                        ("제공목적", row["SUPPORT_COMMENT"]),
                    ),
                },
            })

    return chunks


# ------------------------------------------------------------------
# 메뉴 2: 외부요청 — 요청 1건 = 1청크
# ------------------------------------------------------------------
def build_menu2_chunks(rows: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for row in rows:
        proc_line = None
        if row["PROC_MENT"] is not None:
            detail = _parts(
                ("제공자", row["PROC_PERSON_NAME"]),
                ("제공일", row["PROC_DTM"]),
                ("제공방식", row["TRNS_MTD"]),
            )
            proc_text = cleanse(row["PROC_MENT"])
            proc_line = f"처리내용: {proc_text}" + (f" ({detail})" if detail else "")

        lines = [
            f"[외부요청 {row['REQUEST_ID']}] {cleanse(row['CRP_NAME'])}({cleanse(row['CRP_DEPT_NAME']) or '부서 미상'})의 "
            f"{row['REQ_PERSON_NAME']}이(가) {row['REQ_DTM']} 요청.",
            _line("제목", row["TITLE"]),
            _line("요청내용", row["REQ_MENT"]),
            proc_line,
            _line("특이사항", row["NOTE"]),
            _line("첨부문서", row["ATT_NAMES"]),
        ]
        chunks.append({
            "text": "\n".join(l for l in lines if l),
            "metadata": {
                "데이터유형": "외부요청",
                "원본ID": row["REQUEST_ID"],
                "사업자번호": row["CRP_NO"],
                "기관명": row["CRP_NAME"],
                "요청자": row["REQ_PERSON_NAME"],
                "제공자": row["PROC_PERSON_NAME"],
                "연도": row["REQ_DTM"][:4] if row["REQ_DTM"] else None,
                "세부항목": _detail_items(
                    ("제목", row["TITLE"]),
                    ("요청자", row["REQ_PERSON_NAME"]),
                    ("요청일", _fmt_date_str(row["REQ_DTM"])),
                    ("요청내용", row["REQ_MENT"]),
                    ("처리내용", row["PROC_MENT"]),
                    ("제공자", row["PROC_PERSON_NAME"]),
                    ("제공일", _fmt_date_str(row["PROC_DTM"])),
                    ("제공방식", row["TRNS_MTD"]),
                    ("특이사항", row["NOTE"]),
                ),
            },
        })
    return chunks


# ------------------------------------------------------------------
# 메뉴 3: 내부자료 — REQUEST_ID로 그룹핑, 글 1건 = 1청크 (댓글 시간순 나열)
# IN_USERID는 보류항목 기본값(생략)에 따라 댓글 라인에 표기하지 않음
# ------------------------------------------------------------------
def build_menu3_chunks(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["REQUEST_ID"], []).append(row)

    chunks: list[dict] = []
    for request_id, group in grouped.items():
        head = group[0]
        lines = [
            f"[내부자료 {request_id}] {cleanse(head['TITLE'])} "
            f"(담당: {head['USERNM']}, 결재: {head['DECIDING_OFFICER'] or '미상'}, 등록: {_fmt_date(head['MAIN_CR_DTM'])})",
            _line("주요내용", head["PROC_MENT"]),
        ]

        comment_lines = []
        for row in group:
            if row["FILE_COMMENT"] is None:
                continue
            att = f" [첨부: {row['ATT_NAMES']}]" if row["ATT_NAMES"] else ""
            comment_lines.append(f"- ({_fmt_date(row['DTL_CR_DTM'])}): {cleanse(row['FILE_COMMENT'])}{att}")

        if comment_lines:
            lines.append("")
            lines.append("댓글:")
            lines.extend(comment_lines)

        comments_value = "\n".join(
            f"({_fmt_date(row['DTL_CR_DTM'])}) {cleanse(row['FILE_COMMENT'])}"
            + (f" [첨부: {row['ATT_NAMES']}]" if row["ATT_NAMES"] else "")
            for row in group
            if row["FILE_COMMENT"] is not None
        )

        chunks.append({
            "text": "\n".join(l for l in lines if l is not None),
            "metadata": {
                "데이터유형": "내부자료",
                "원본ID": request_id,
                "담당자": head["USERNM"],
                "결재자": head["DECIDING_OFFICER"],
                "연도": str(head["MAIN_CR_DTM"].year) if head["MAIN_CR_DTM"] else None,
                "세부항목": _detail_items(
                    ("제목", head["TITLE"]),
                    ("담당자", head["USERNM"]),
                    ("결재자", head["DECIDING_OFFICER"]),
                    ("등록일", _fmt_date(head["MAIN_CR_DTM"])),
                    ("주요내용", head["PROC_MENT"]),
                    ("댓글", comments_value or None),
                ),
            },
        })
    return chunks


# ------------------------------------------------------------------
# 메뉴 4: 심의위원회 — 안건 1건 = 1청크
# ------------------------------------------------------------------
def build_menu4_chunks(rows: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for row in rows:
        # AGENDA_NUMBER가 NULL이면 f-string이 그대로 "None" 문자열을 삽입해버리는 버그가
        # 있었다(2026-10-20 발견 — 전체 58건 중 11건에서 재현, "위원회 안건 None —"처럼
        # 뜬금없는 "None"이 본문에 그대로 노출됨). 사실관계 왜곡은 아니지만 순수한 표기
        # 품질 문제라 정리 — 값이 있을 때만 그 뒤에 붙인다. (참고: 처음엔 이게 같은 날
        # 발견한 "한국은행 1차/3차 위원회" 답변 불안정의 원인일 수 있다고 의심해 먼저
        # 고쳤으나, 수정 후에도 재현율이 그대로(15/15 실패)라 무관함을 확인 — 진짜 원인은
        # 별도로 찾은 답변생성 history anchoring 문제였다. 이 수정 자체는 독립적으로
        # 정당한 데이터 품질 개선이라 그대로 유지.)
        agenda_number = row.get("AGENDA_NUMBER")
        agenda_suffix = f" {agenda_number}" if agenda_number not in (None, "") else ""
        lines = [
            f"[심의안건 {row['DISCUSSION_ID']}] {row['YEAR']}년 {row['SEQ']}차 위원회 안건{agenda_suffix} — "
            f"{cleanse(row['CRP_NAME'])}({cleanse(row['CRP_DEPT_NAME']) or '부서 미상'})",
            _line("제목", row["TITLE"]),
            _line("요청내용", row["REQ_MENT"]),
            _line("법적근거", row["LAW_COMMENT"]),
            _line("요청항목", row["REQ_ITEM"]),
            _line("심의결과", row["DISCUSSION_RESULT"]),
            _line("결정내용", row["RESULT_CONTENT"]),
            _line("제공항목", row["PROC_ITEM"]),
            _line("특이사항", row["NOTE"]),
            _line("첨부문서", row["ATT_NAMES"]),
        ]
        chunks.append({
            "text": "\n".join(l for l in lines if l),
            "metadata": {
                "데이터유형": "심의위원회",
                "원본ID": row["DISCUSSION_ID"],
                "사업자번호": row["CRP_NO"],
                "기관명": row["CRP_NAME"],
                "연도": row["YEAR"],
                "차수": row["SEQ"],
                "심의결과": _normalize_result(row["DISCUSSION_RESULT"]),
                "세부항목": _detail_items(
                    ("제목", row["TITLE"]),
                    ("요청내용", row["REQ_MENT"]),
                    ("법적근거", row["LAW_COMMENT"]),
                    ("요청항목", row["REQ_ITEM"]),
                    ("심의결과", row["DISCUSSION_RESULT"]),
                    ("결정내용", row["RESULT_CONTENT"]),
                    ("제공항목", row["PROC_ITEM"]),
                    ("특이사항", row["NOTE"]),
                ),
            },
        })
    return chunks
