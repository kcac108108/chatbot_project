"""2단계: 첨부파일 내용을 청킹·임베딩해서 벡터DB에 반영한다.
메뉴(DOCUCD)별로 단계적으로 롤아웃. 직접추출 가능한 형식(엑셀·한글·텍스트 등)과 비전모델이
필요한 형식(스캔본 PDF·이미지, zip 내부 포함) 모두 `attachment_extract.extract_any`가
알아서 구분해 처리한다."""
import re
import uuid
from pathlib import Path

import tiktoken
from pydantic import BaseModel
from qdrant_client.http import models as qmodels

from app.core.config import ATTACHMENT_BASE_DIR
from app.core.config import DB_TABLE_SUFFIX as _SFX
from app.core.db import fetch_all
from app.core.openai_client import chat_parse, embed_texts
from app.core.vectordb import ensure_collection, find_by_type_and_source, upsert_points
from app.services.attachment_extract import extract_any

# DOCUCD -> (부모 테이블, 부모ID 컬럼, 이 첨부가 속하는 첨부전용 데이터유형)
# 데이터유형을 메뉴별로 나눠서(예: "외부요청첨부") 기존 메뉴 스코프 필터링 구조에
# 그대로 편입되게 한다 — "첨부내용" 하나로 뭉치면 메뉴 스코프별 검색 범위 제어가 안 됨.
_DOCUCD_MAP = {
    "C0002010": ("HISTORY_ORGANIZATION_INFO", "ORGANIZATION_ID", "기관정보첨부"),
    "C0002011": ("HISTORY_ORGANIZATION_DETAIL", "DETAIL_ID", "신청이력첨부"),
    "C0002012": ("HISTORY_EXTERNAL_REQUEST", "REQUEST_ID", "외부요청첨부"),
    "C0002013": ("HISTORY_DISCUSSION_COMMITTEE", "DISCUSSION_ID", "심의위원회첨부"),
    "C0002014": ("HISTORY_INTERNAL_INFO", "REQUEST_ID", "내부자료첨부"),
}

# 1~4번 메뉴는 청크가 수백 자라 BATCH_SIZE=100이 안전했지만, 첨부 청크는 훨씬 길어서
# 그대로 쓰면 OpenAI 요청 한도(30만 토큰/요청)를 넘길 수 있음(실제로 재현됨) — 작게 잡는다.
BATCH_SIZE = 20

# text-embedding-3-large 한도(8,191토큰) 대비 여유 있게(헤더 텍스트가 더해지는 것 감안).
# 애초에 글자 수(6,000자) 기준으로 나눴다가, 실제 실행 중 한 청크가 8,192토큰 한도를
# 넘겨 배치 임베딩이 통째로 실패하는 걸 발견함(2026-09-21) — 한글/숫자·기호 비중에 따라
# 글자당 토큰 수가 들쭉날쭉해서 글자 수 기준은 안전을 보장 못함. tiktoken으로 실제
# 토큰 수를 세어 나누는 방식으로 교체.
_CHUNK_TOKEN_LIMIT = 7500
_TOKENIZER = tiktoken.encoding_for_model("text-embedding-3-large")


# 관리자 확인(2026-09-02, 2026-09-03 방식 변경): DB 테이블/컬럼 카탈로그성 자료(적재테이블
# 목록, 컬럼 정의서 등)·SQL 쿼리 파일은 실제 이력 서술이 아니라 시스템/기술 정보라 정보가치가
# 낮다고 판단됨. 처음엔 완전히 숨겼으나(2026-09-02), 이후 "관련첨부 목록엔 그대로 노출하되
# 내용 대신 파일 저장 경로만 안내"하는 방식으로 확정(2026-09-03) — 존재는 알 수 있어야
# 필요하면 시스템에서 직접 찾아볼 수 있음. .sql은 확장자로 자동 판별, 카탈로그성 파일은
# 지금은 파일명으로 직접 골라내는 수동 방식 — 대상이 많아지는 카테고리에서는 자동 분류로 바꿔야 함.
_PATH_ONLY_FILENAMES = {
    "(20230907)관세청_적재테이블_리스트.xls",
    "(20230718)CS 검사 관련 컬럼(최종).xls",
    # 내부자료(C0002014) 검토 중 발견 — 항목정의서는 DB 컬럼 정의(25만자), Analytic Function은
    # 이 사건과 무관한 일반 SQL 기술 튜토리얼(2026-09 확인)
    "한국무역통계진흥원 자사실적정보 4G 항목정의서_20231114_업데이트중.xlsx",
    "Analytic Function.doc",
    # 신청이력첨부·내부자료 엑셀 재검토(2026-09-18) — 사용자 판단으로 제외 확정.
    # "목록통관+수출목록 변환신고"만 DB 컬럼정의 카탈로그(TABLE_NAME/COLUMN_NAME 패턴)이고
    # 나머지 3건은 실제 업무 데이터였으나, 사용자가 챗봇 정보가치 기준으로 함께 제외 판단함.
    "TIPA 분기별 금액리스트(2022년).xlsx",
    "TIPA 분기별 금액리스트(2021년).xlsx",
    "효성계열사 전송 오류 확인 및 재전송_20240923.xlsx",
    "목록통관+수출목록 변환신고.xlsx",
}


def _is_path_only(filename: str) -> bool:
    return filename in _PATH_ONLY_FILENAMES or filename.lower().endswith(".sql")


def _valid_attachments(docucd: str, since=None) -> list[dict]:
    """since를 주면 그 시각 이후 등록·수정된 첨부만 반환한다(신규 이력 자동처리 전용,
    2026-09-29) — 기존 이력에 나중에 파일이 추가되는 경우도 잡아야 해서 부모 레코드가
    아니라 TM_ATTFILE 자신의 등록·수정일시(CR_DTM/IN_DTM)를 기준으로 본다. CR_DTM/IN_DTM도
    같이 반환해서(2026-10-12), 결과 파일이 아직 안 생긴 건을 다음 실행에 재시도할 때
    기준시각을 어디까지 되돌릴지 계산할 수 있게 한다."""
    table, col, _ = _DOCUCD_MAP[docucd]
    since_clause = "AND (A.CR_DTM > :since OR A.IN_DTM > :since)" if since else ""
    return fetch_all(
        f"""
        SELECT A.ATTFILE_ID, A.ATTFILE_NUM, A.ATTFILE_NM, A.FILE_PATH, A.CR_DTM, A.IN_DTM
        FROM TM_ATTFILE{_SFX} A
        WHERE A.DOCUCD = :docucd
          AND EXISTS (SELECT 1 FROM {table}{_SFX} T WHERE T.{col} = A.ATTFILE_ID)
          {since_clause}
        ORDER BY A.ATTFILE_ID, A.ATTFILE_NUM
        """,
        {"docucd": docucd, "since": since} if since else {"docucd": docucd},
    )


def _local_path(file_path_db: str, docucd: str, attfile_id: str) -> Path:
    filename = file_path_db.split("/")[-1]
    return Path(ATTACHMENT_BASE_DIR) / docucd / attfile_id / filename


def _result_text_path(file_path_db: str, docucd: str, attfile_id: str) -> Path:
    """전산팀 야간 배치가 만드는 OCR/추출 결과 파일 경로(2026-10-12) — 원본과 같은
    폴더·같은 파일명, 확장자만 `.txt`로 바뀐다(스캔본·엑셀·한글·zip 등 형식 무관하게
    전산팀이 전부 이 규칙으로 결과를 생성해줌, zip은 내부 파일들을 합쳐 결과 1개).
    전산팀 OCR API를 우리가 직접 호출하던 기존 방식(`ocr_remote.py`)은 이걸로 대체됨."""
    return _local_path(file_path_db, docucd, attfile_id).with_suffix(".txt")


def _review_filename(att: dict, path: Path) -> str:
    return f"{att['ATTFILE_ID']}__{att['ATTFILE_NUM']}__{path.stem}.txt"


# OCR/비전모델 추출 결과에 페이지 하단 쪽번호("- 1 -", "- 12 -" 등)가 그대로 한 줄로
# 섞여 나오는 경우가 실측으로 아주 흔함(2026-09-21, 전체 3,135개 청크 중 2,022줄에서
# 발견) — 실제 내용이 아니라 순수 잡음이라 안전하게 제거 가능.
_PAGE_NUMBER_LINE_RE = re.compile(r"^[ \t]*-\s*\d{1,4}\s*-[ \t]*$", re.MULTILINE)


def _strip_page_numbers(text: str) -> str:
    cleaned = _PAGE_NUMBER_LINE_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def _split_chunks(text: str, limit: int = _CHUNK_TOKEN_LIMIT) -> list[str]:
    tokens = _TOKENIZER.encode(text)
    if len(tokens) <= limit:
        return [text]
    return [_TOKENIZER.decode(tokens[i : i + limit]) for i in range(0, len(tokens), limit)]


def _point_id(data_type: str, source_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{data_type}:{source_id}"))


class _AttachmentSummary(BaseModel):
    요약: str


# 요약용 입력은 넉넉히 크게 잡되, 모델 컨텍스트 한도 안에 안전하게 들어오도록 제한한다.
# (실제로 40만자짜리 파일이 있어 전문을 그대로 넣으면 한도를 넘길 수 있음)
_SUMMARIZE_INPUT_LIMIT = 12000

_SUMMARIZE_PROMPT_BASE = (
    "다음은 첨부파일에서 추출한 원문 내용이다. 챗봇 화면에 보여줄 간략한 요약으로 정리하라.\n"
    "- [이 건의 기본 정보]가 함께 주어지면, 그 기관이 왜/무엇 때문에 이 자료를 "
    "요청·제공했는지(신청목적·제공목적 등)부터 요약 맨 앞에서 먼저 짚어라 — 이게 가장 "
    "중요한 맥락이다\n"
    "- 반드시 포함: 구체적 수치(건수·금액 등), 배경/근거(지침·법령 개정 등 왜 이 내용이 생겼는지), 적용 기간·조건\n"
    "- 담당자·결재자·문서번호 같은 행정 정보는 핵심이 아니므로 굳이 언급하지 마라\n"
    "- 항목 나열이 아니라 자연스러운 문장으로 서술\n"
    "- [이 건의 기본 정보]에 없는 내용은 반드시 원문에서만 가져오고, 원문에도 없는 "
    "내용을 지어내지 마라\n"
)

# 데이터유형(첨부) -> 부모 레코드의 데이터유형. chat_service.py의 _BASE_TYPE_OF와
# 같은 매핑이지만, 순환 import를 피하려고(attachment_ingest가 chat_service를
# 끌어오게 됨) 이 모듈에 독립적으로 둔다.
_ATTACHMENT_BASE_TYPE_OF = {
    "기관정보첨부": "기관정보",
    "신청이력첨부": "신청이력",
    "외부요청첨부": "외부요청",
    "내부자료첨부": "내부자료",
    "심의위원회첨부": "심의위원회",
}


def _attachment_parent_context(data_type: str, parent_id: str) -> str:
    """첨부파일 원문만으로는 "이 자료가 왜/무엇 때문에 요청됐는지" 같은 가장 중요한
    맥락이 아예 안 담길 수 있다(원문 PDF/계약서 자체엔 그런 문구가 없고, 신청목적·
    제공목적 등은 DB 정형 컬럼에만 있는 정보라서) — 2026-10-21 사용자 지적: "요약에
    저 업체가 뭐 때문에 요청했는지가 하나도 없다". 부모 레코드의 세부항목(DB에서 이미
    정확히 뽑아놓은 라벨:값 목록)을 그대로 프롬프트 맥락으로 제공해, LLM이 원문 밖에서
    지어내지 않고도 이 핵심 정보를 요약에 포함할 수 있게 한다."""
    base_type = _ATTACHMENT_BASE_TYPE_OF.get(data_type)
    if not base_type:
        return ""
    parents = find_by_type_and_source(base_type, parent_id, limit=1)
    if not parents:
        return ""
    parent = parents[0]
    lines = []
    if parent.get("기관명"):
        lines.append(f"기관명: {parent['기관명']}")
    for field in parent.get("세부항목") or []:
        if field.get("value"):
            lines.append(f"{field['label']}: {field['value']}")
    if not lines:
        return ""
    return "[이 건의 기본 정보]\n" + "\n".join(lines) + "\n\n"

# 원문 길이가 길수록 담아야 할 정보(사건·수치·조건)도 많아지므로, 요약 분량도 원문 길이에
# 비례해서 늘어나도록 구간을 나눈다 — 짧은 문서를 억지로 늘리거나 긴 문서를 3~5문장으로
# 눌러 담아 정보가 유실되는 것을 방지(사용자 피드백, 2026-09-18).
_SUMMARIZE_LENGTH_TIERS = [
    (500, "1~2문장, 80~150자 내외로 작성"),
    (2000, "2~3문장, 150~300자 내외로 작성"),
    (5000, "3~5문장, 200~400자 내외로 작성"),
    (9000, "5~8문장, 400~700자 내외로 작성"),
    (float("inf"), "8~12문장, 700~1000자 내외로 작성"),
]


def _summarize_length_guide(length: int) -> str:
    for threshold, guide in _SUMMARIZE_LENGTH_TIERS:
        if length <= threshold:
            return guide
    return _SUMMARIZE_LENGTH_TIERS[-1][1]


def _summarize_attachment(text: str, filename: str, parent_context: str = "") -> str:
    """첨부파일 원문 전체를 그대로 보여주면 '번잡스럽고 정리 안 된' 느낌이라는 사용자
    피드백에 따라, 적재 시점에 핵심만 추린 요약을 미리 만들어 저장해둔다. 요약 분량은
    원문 길이에 맞춰 조절한다(짧은 문서=짧은 요약, 긴 문서=더 여유로운 요약)."""
    truncated = text if len(text) <= _SUMMARIZE_INPUT_LIMIT else text[:_SUMMARIZE_INPUT_LIMIT]
    prompt = _SUMMARIZE_PROMPT_BASE + f"- {_summarize_length_guide(len(truncated))}"
    result = chat_parse(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"{parent_context}[파일명: {filename}]\n{truncated}"},
        ],
        response_format=_AttachmentSummary,
    )
    return result.요약


def extract_for_review(docucd: str, out_dir: Path) -> list[dict]:
    """1단계: 임베딩 전에 텍스트만 추출해서 파일로 저장 — 사람이 먼저 확인하는 용도.
    직접추출·비전모델 모두 이 단계에서 처리되므로, 비전모델 대상 파일은 실제 API 호출이
    발생한다(비용 발생) — 소량으로 먼저 돌려보고 검토할 것."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for att in _valid_attachments(docucd):
        row = {**att}
        if _is_path_only(att["ATTFILE_NM"]):
            row["상태"] = "경로전용(테이블/SQL 정보 — 내용 미추출)"
            results.append(row)
            continue
        path = _local_path(att["FILE_PATH"], docucd, att["ATTFILE_ID"])
        if not path.exists():
            row["상태"] = "파일없음"
        else:
            text = extract_any(path)
            if not text or not text.strip():
                row["상태"] = "추출결과없음"
            else:
                out_name = _review_filename(att, path)
                (out_dir / out_name).write_text(text, encoding="utf-8")
                row["상태"] = "추출완료"
                row["글자수"] = len(text)
                row["저장파일"] = out_name
        results.append(row)
    return results


def _build_entries_for_text(
    parent_id: str, att_num, filename: str, text: str, data_type: str, file_path: str | None = None
) -> list[tuple[str, dict]]:
    """추출된 원문 텍스트 하나를 (쪽번호 제거→요약→토큰 기준 분할→payload 조립)까지
    처리해 Qdrant에 넣을 (source_id, payload) 목록으로 만든다. 수동 파이프라인
    (ingest_attachments)과 신규 이력 자동처리 파이프라인이 공용으로 쓴다.

    file_path(TM_ATTFILE.FILE_PATH)를 payload에 같이 저장해두면, 경로전용(SQL/카탈로그성)
    첨부뿐 아니라 내용이 있는 일반 첨부도 원본 파일을 다운로드할 수 있게 된다
    (2026-11-05 — 그 전까지는 경로전용 첨부에만 파일경로가 저장돼 있었음)."""
    text = _strip_page_numbers(text)
    parent_context = _attachment_parent_context(data_type, parent_id)
    summary = _summarize_attachment(text, filename, parent_context)
    chunks = _split_chunks(text)
    entries: list[tuple[str, dict]] = []
    for idx, chunk_text in enumerate(chunks):
        source_id = f"{parent_id}:{att_num}:{idx}"
        suffix = f" ({idx + 1}/{len(chunks)}번째 조각)" if len(chunks) > 1 else ""
        body = f"[{data_type} {parent_id} / {filename}]{suffix}\n{chunk_text}"
        payload = {
            "text": body,
            "데이터유형": data_type,
            "원본ID": parent_id,
            "첨부파일명": filename,
            "요약": summary,
        }
        if file_path:
            payload["파일경로"] = file_path
        entries.append((source_id, payload))
    return entries


def _upsert_entries(data_type: str, entries: list[tuple[str, dict]], log_prefix: str) -> int:
    """(source_id, payload) 목록을 배치로 임베딩해 Qdrant에 upsert. 수동·자동 파이프라인 공용."""
    ensure_collection()
    total = 0
    for i in range(0, len(entries), BATCH_SIZE):
        batch = entries[i : i + BATCH_SIZE]
        vectors = embed_texts([payload["text"] for _, payload in batch])
        points = [
            qmodels.PointStruct(id=_point_id(data_type, source_id), vector=vector, payload=payload)
            for (source_id, payload), vector in zip(batch, vectors)
        ]
        upsert_points(points)
        total += len(points)
        print(f"[{log_prefix}] {total}/{len(entries)} 청크 적재 완료")
    return total


def ingest_attachments(docucd: str, review_dir: Path | None = None) -> int:
    """2단계: 첨부 전체(직접추출+비전모델)를 실제로 청킹+임베딩+Qdrant 반영.

    review_dir을 주면, 비전모델 재호출 대신 그 폴더의 검토 파일(extract_for_review가
    만든 것과 같은 이름 규칙)을 우선 읽는다 — 비전모델 결과는 표 구조 등에서 실수가 있을
    수 있어, 사람이 검토 파일을 먼저 확인·수정해두면 그 수정 내용이 그대로 반영되고
    재추출로 인한 결과 변동(같은 이미지라도 매번 다르게 나올 수 있음)도 피할 수 있다.
    검토 파일이 없으면 그때만 비전모델/직접추출을 새로 실행한다."""
    ensure_collection()
    _, _, data_type = _DOCUCD_MAP[docucd]

    entries: list[tuple[str, dict]] = []  # (point 식별용 source_id, payload)
    for att in _valid_attachments(docucd):
        parent_id = att["ATTFILE_ID"]
        filename = att["ATTFILE_NM"]

        if _is_path_only(filename):
            source_id = f"{parent_id}:{att['ATTFILE_NUM']}:0"
            payload = {
                "text": f"[{data_type} {parent_id} / {filename}] (테이블/SQL 정보 파일 - 내용 미제공)",
                "데이터유형": data_type,
                "원본ID": parent_id,
                "첨부파일명": filename,
                "경로전용": True,
                "파일경로": att["FILE_PATH"],
            }
            entries.append((source_id, payload))
            print(f"[attachment_ingest:{docucd}] 경로만 표시: {filename}")
            continue

        path = _local_path(att["FILE_PATH"], docucd, parent_id)
        review_path = Path(review_dir) / _review_filename(att, path) if review_dir else None
        if review_path and review_path.exists():
            text = review_path.read_text(encoding="utf-8")
            print(f"[attachment_ingest:{docucd}] 검토파일 사용({len(text)}자): {filename}")
        elif not path.exists():
            print(f"[attachment_ingest:{docucd}] 파일없음: {filename}")
            continue
        else:
            text = extract_any(path)
            if not text or not text.strip():
                print(f"[attachment_ingest:{docucd}] 추출결과없음: {filename}")
                continue
            print(f"[attachment_ingest:{docucd}] 추출완료({len(text)}자): {filename}")

        entries += _build_entries_for_text(
            parent_id, att["ATTFILE_NUM"], filename, text, data_type, file_path=att["FILE_PATH"]
        )

    return _upsert_entries(data_type, entries, f"attachment_ingest:{docucd}")
