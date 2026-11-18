import asyncio
import logging
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.core.config import ATTACHMENT_BASE_DIR, CHAT_REQUEST_TIMEOUT_SECONDS
from app.models.schemas import (
    AttachmentDetailResponse,
    ChatRequest,
    ChatResponse,
    KeywordSearchResponse,
    MenuScope,
    OrganizationHistoryResponse,
    SourceItem,
)
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


# "빠른 검색" — 사이드바 입력창에서 사용자가 직접 입력한 단어를 자연어 질문분석 없이
# 그대로 전수 스캔해 매칭된 자료를 보여준다(2026-11-12). LLM 호출(질문분석·답변생성)이
# 전혀 없어 /chat과 달리 전체 제한시간을 걸 필요가 없다 — Qdrant 스캔만 하는 가벼운 요청.
@router.get("/keyword-search", response_model=KeywordSearchResponse)
def keyword_search(keyword: str, menu_scope: MenuScope | None = None) -> KeywordSearchResponse:
    return chat_service.keyword_search(keyword, menu_scope)


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


# 첨부 미리보기 밑 "상세" 버튼 — 사전 생성 요약보다 더 자세한 설명을 사용자가 클릭한
# 그 순간에 실시간으로 생성한다(요청마다 GPT 호출 발생, 호출 빈도가 낮아 비용 문제 없음).
@router.get("/attachment-detail", response_model=AttachmentDetailResponse)
def attachment_detail(data_type: str, source_id: str, filename: str) -> AttachmentDetailResponse:
    detail = chat_service.generate_attachment_detail(data_type, source_id, filename)
    return AttachmentDetailResponse(상세=detail)


# 내용은 안 보여주기로 한 첨부(테이블/SQL 정보성 자료)를 대신 다운로드할 수 있게 하는
# 엔드포인트. DB에 저장된 경로(예: /attfiles/C0002012/H20240100178088/xxx.sql) 형식만
# 정확히 허용해 임의 경로 접근(path traversal)을 막는다 — 사용자가 준 경로를 그대로 믿지
# 않고, DOCUCD/ATTFILE_ID/파일명 세 조각으로만 재조립해 로컬 경로를 만든다.
_FILE_PATH_RE = re.compile(r"^/attfiles/([A-Za-z0-9]+)/([A-Za-z0-9]+)/([^/\\]+)$")


@router.get("/attachment-download")
def download_attachment(file_path: str, display_name: str | None = None) -> FileResponse:
    match = _FILE_PATH_RE.match(file_path)
    if not match:
        raise HTTPException(400, "잘못된 파일 경로입니다")
    docucd, attfile_id, filename = match.groups()
    local_path = Path(ATTACHMENT_BASE_DIR) / docucd / attfile_id / filename
    if not local_path.exists():
        raise HTTPException(404, "파일을 찾을 수 없습니다")
    # local_path(물리 저장 파일명, 예: 1727073212442_2.xlsx)를 그대로 내려주면 사용자가
    # 받은 파일을 알아볼 수 없다(2026-10-13 테스트 중 발견) — 화면에 이미 표시 중인 원본
    # 파일명(display_name)이 있으면 그걸 다운로드 파일명으로 쓴다. display_name은 경로
    # 조립에 전혀 쓰이지 않으므로(local_path는 이미 위에서 안전하게 확정됨) 검증 없이
    # 그대로 받아도 path traversal 위험이 없다 — 확장자가 원본과 다르면(손상된 값 등)
    # 안전하게 물리 파일명으로 폴백한다.
    download_name = filename
    if display_name and Path(display_name).suffix.lower() == Path(filename).suffix.lower():
        download_name = display_name
    return FileResponse(local_path, filename=download_name)


# ⚠️ 임시 진단용(2026-11-17) — 장명근 피드백 "메뉴별 대화 메모리" 가능 여부를 검토하다가,
# 로그인 없이 IP로라도 사용자를 구분하는 방안이 나왔는데 그 전제(아파치 리버스 프록시가
# 실제 클라이언트 IP를 우리 쪽에 넘겨주는지)부터 확인이 필요해서 추가함. 읽기 전용,
# 데이터 변경·저장 전혀 없음 — 서로 다른 PC에서 열어봐서 값이 PC별로(그 PC의 실제 사설
# IP와 일치하게) 다르게 나오면 정상 전달되는 것. 확인 끝나면 지워도 무방.
@router.get("/debug-client-ip")
def debug_client_ip(request: Request) -> dict:
    return {
        "x_forwarded_for": request.headers.get("x-forwarded-for"),
        "x_real_ip": request.headers.get("x-real-ip"),
        "client_host": request.client.host if request.client else None,
    }
