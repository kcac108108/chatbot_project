from fastapi import APIRouter

from app.models.schemas import ChatRequest, ChatResponse
from app.services import chat_service

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return chat_service.ask(request.question, request.session_id)
