import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import ATTACHMENT_BASE_DIR
from app.models.schemas import ChatRequest, ChatResponse, OrganizationHistoryResponse
from app.services import chat_service

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return chat_service.ask(request.question, request.session_id, request.menu_scope)


# 기관정보 카드의 "이력 보기" 버튼 — 그 기관의 신청이력 전체를 top-K 제한 없이, LLM 서술
# 없이 코드가 직접 날짜+제목+원본ID 표로 반환한다.
@router.get("/organization-history", response_model=OrganizationHistoryResponse)
def organization_history(organization_id: str) -> OrganizationHistoryResponse:
    return OrganizationHistoryResponse(items=chat_service.get_organization_history(organization_id))


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
