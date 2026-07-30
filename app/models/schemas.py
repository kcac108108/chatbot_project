from typing import Literal, Optional

from pydantic import BaseModel, Field

DATA_TYPES = Literal["기관정보", "신청이력", "외부요청", "내부자료", "심의위원회"]


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class SourceItem(BaseModel):
    데이터유형: str
    원본ID: str
    기관명: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceItem]


# ------------------------------------------------------------------
# 내부 LLM 구조화 출력 스키마 (정의서 5장 확정 메타 필드 기준)
# ------------------------------------------------------------------
class FilterExtraction(BaseModel):
    데이터유형: Optional[DATA_TYPES] = None
    기관명: Optional[str] = None
    연도: Optional[str] = Field(None, description="4자리 연도 문자열, 예: 2024")
    통계종류: Optional[str] = None
    수출입구분: Optional[str] = None
    요청자: Optional[str] = None
    제공자: Optional[str] = None
    담당자: Optional[str] = None
    결재자: Optional[str] = None
    심의결과: Optional[str] = None
    차수: Optional[int] = None


class QueryAnalysis(BaseModel):
    standalone_question: str = Field(description="이전 대화 맥락을 반영해 이 질문만으로 검색 가능하도록 다시 쓴 독립형 질문")
    filters: FilterExtraction


class GroundedAnswer(BaseModel):
    answer: str
    used_indices: list[int] = Field(default_factory=list, description="답변 근거로 실제 사용한 자료 번호(1-based)")
