import asyncio
import logging
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import ATTACHMENT_BASE_DIR, CHAT_REQUEST_TIMEOUT_SECONDS
from app.models.schemas import ChatRequest, ChatResponse, OrganizationHistoryResponse, SourceItem
from app.services import chat_service

router = APIRouter()
logger = logging.getLogger(__name__)


# OpenAI 호출이 드물게 매우 느려지거나 멈추는 경우를 대비해 요청 전체에 제한시간을 둔다
# (2026-09-28 실제 1시간 넘게 응답이 안 오는 요청을 관찰함). session_id는 ask() 내부가
# 아니라 여기서 먼저 확정해둬야, 타임아웃으로 ask()가 끝까지 못 가도 같은 session_id로
# 안전한 안내를 돌려줄 수 있다. 타임아웃이 나도 백그라운드 스레드의 실제 OpenAI 호출
# 자체는 계속 진행된다(파이썬 스레드는 강제 종료 불가) — 이 요청의 응답만 포기하는 것.
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid.uuid4())
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(chat_service.ask, request.question, session_id, request.menu_scope),
            timeout=CHAT_REQUEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("chat 요청 타임아웃(%s초 초과): session_id=%s", CHAT_REQUEST_TIMEOUT_SECONDS, session_id)
        return ChatResponse(session_id=session_id, answer=chat_service.REQUEST_TIMEOUT_ANSWER, sources=[])


# 기관정보 카드의 "이력 보기" 버튼 — 그 기관의 신청이력 전체를 top-K 제한 없이, LLM 서술
# 없이 코드가 직접 날짜+제목+원본ID 표로 반환한다.
@router.get("/organization-history", response_model=OrganizationHistoryResponse)
def organization_history(organization_id: str) -> OrganizationHistoryResponse:
    return OrganizationHistoryResponse(items=chat_service.get_organization_history(organization_id))



# 이력 보기 표의 한 행을 클릭했을 때 — 그 건 하나만 메인 근거자료 카드와 같은 형태
# (세부항목 표+관련첨부)로 상세 조회한다.
@router.get("/record-detail", response_model=SourceItem)
def record_detail(data_type: str, source_id: str) -> SourceItem:
    detail = chat_service.get_record_detail(data_type, source_id)
    if detail is None:
        raise HTTPException(404, "레코드를 찾을 수 없습니다")
    return detail


# 내용은 안 보여주기로 한 첨부(테이블/SQL 정보성 자료)를 대신 다운로드할 수 있게 하는
# 엔드포인트. DB에 저장된 경로(예: /attfiles/C0002012/H20240100178088/xxx.sql) 형식만
# 정확히 허용해 임의 경로 접근(path traversal)을 막는다 — 사용자가 준 경로를 그대로 믿지
# 않고, DOCUCD/ATTFILE_ID/파일명 세 조각으로만 재조립해 로컬 경로를 만든다.
_FILE_PATH_RE = re.compile(r"^/attfiles/([A-Za-z0-9]+)/([A-Za-z0-9]+)/([^/\\]+)$")


@router.get("/attachment-download")
def download_attachment(file_path: str) -> FileResponse:
    match = _FILE_PATH_RE.match(file_path)
    if not match:
        raise HTTPException(400, "잘못된 파일 경로입니다")
    docucd, attfile_id, filename = match.groups()
    local_path = Path(ATTACHMENT_BASE_DIR) / docucd / attfile_id / filename
    if not local_path.exists():
        raise HTTPException(404, "파일을 찾을 수 없습니다")
    return FileResponse(local_path, filename=filename)
