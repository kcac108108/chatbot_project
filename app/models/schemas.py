from typing import Literal, Optional

from pydantic import BaseModel, Field

DATA_TYPES = Literal["기관정보", "신청이력", "외부요청", "내부자료", "심의위원회"]


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    menu_scope: Optional[Literal["기관별 사업정보", "외부요청", "내부자료", "심의위원회", "첨부파일"]] = Field(
        None, description="사용자가 웹 화면에서 특정 메뉴로 범위를 고정한 경우. 미지정(전체)이면 null."
        "'첨부파일'은 메뉴가 아니라 5개 메뉴의 첨부내용 전체를 가로질러 검색하는 전용 범위"
    )


class DetailField(BaseModel):
    label: str
    value: str


class RelatedAttachment(BaseModel):
    첨부파일명: str
    미리보기: str
    파일경로: Optional[str] = None


class HistoryItem(BaseModel):
    원본ID: str
    신청일: Optional[str] = None
    사업명: Optional[str] = None


class OrganizationHistoryResponse(BaseModel):
    items: list[HistoryItem] = Field(default_factory=list)


class SourceItem(BaseModel):
    데이터유형: str
    원본ID: str
    기관명: Optional[str] = None
    미리보기: Optional[str] = None
    세부항목: list[DetailField] = Field(default_factory=list)
    관련첨부: list[RelatedAttachment] = Field(default_factory=list)


class AnswerDetail(BaseModel):
    label: str
    narrative: str
    source: Optional[SourceItem] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    details: list[AnswerDetail] = Field(default_factory=list)
    sources: list[SourceItem]


# ------------------------------------------------------------------
# 내부 LLM 구조화 출력 스키마 (정의서 5장 확정 메타 필드 기준)
# ------------------------------------------------------------------
class FilterExtraction(BaseModel):
    데이터유형: Optional[DATA_TYPES] = Field(
        None,
        description="질문에 언급된 메뉴/데이터종류를 매핑하라 — 기관정보(기관별 사업정보 메뉴),"
        "신청이력(기관별 사업정보의 신청 이력), 외부요청(외부요청 메뉴), 내부자료(내부자료 메뉴),"
        "심의위원회(심의위원회 메뉴). 명시적으로 언급 안 됐으면 null",
    )
    기관명: Optional[str] = None
    연도: Optional[str] = Field(None, description="4자리 연도 문자열, 예: 2024")
    통계종류: Optional[str] = Field(
        None,
        description="기관별 사업정보(메뉴1) 이력에만 존재하는 필드다. 심의위원회 등 다른 메뉴에는"
        "이 필드 자체가 없으니 절대 채우지 마라. '3차'처럼 차수를 가리키는 표현은 여기가 아니라"
        "차수 필드에 넣어라 — 통계종류와 차수에 같은 표현을 중복해서 채우지 마라",
    )
    수출입구분: Optional[str] = Field(
        None, description="기관별 사업정보(메뉴1) 이력에만 존재하는 필드다. 다른 메뉴에는 채우지 마라"
    )
    요청자: Optional[str] = Field(None, description="질문에서 '요청했다/신청했다'처럼 요청자 역할로 명시된 사람만. 다른 역할로 언급된 이름을 넣지 마라")
    제공자: Optional[str] = Field(None, description="질문에서 '제공했다/처리했다'처럼 제공자 역할로 명시된 사람만. 다른 역할로 언급된 이름을 넣지 마라")
    담당자: Optional[str] = Field(None, description="질문에서 '담당했다/맡았다'처럼 담당자 역할로 명시된 사람만. 다른 역할로 언급된 이름을 넣지 마라")
    결재자: Optional[str] = Field(None, description="질문에서 '결재했다/승인했다'처럼 결재자 역할로 명시된 사람만. 다른 역할로 언급된 이름을 넣지 마라")
    심의결과: Optional[str] = None
    차수: Optional[int] = None
    키워드: Optional[str] = Field(
        None,
        description="질문에 나온 단어가 자료 본문에 실제로 포함된 것을 빠짐없이 찾아야 할 때만 채워라"
        "(예: '국가'). 위 다른 필드(기관명·통계종류·수출입구분·담당자 등)로 이미 표현되는 개념이면"
        "여기 채우지 마라 — 그런 구조화 필드로 커버 안 되는 애매한 주제어·키워드일 때만 사용",
    )
    # 2026-10-20 추가 — "2024-01-01 이후에 등록된 이력 있어?" 류 날짜 임계값 질문은 검색
    # 자체(필터+리트리벌)는 항상 정확한데, 자료가 그 날짜 조건을 만족하는지 "비교"하는 걸
    # 답변 생성 LLM에게 맡기면 신뢰도가 떨어짐(특히 기준일에 가까운 날짜일수록 — "12월 1일
    # 이후" vs 레코드의 "12월 27일" 비교가 가끔 틀림, 반복 측정 결과 2/10까지 떨어짐). 날짜
    # "파싱"(이 질문이 가리키는 기준 날짜가 몇 월 며칠인지 읽어내는 것)은 LLM이 안정적으로
    # 잘 하지만, 날짜 "비교"(그 기준과 실제 레코드 날짜 중 어느 게 더 이른지)는 안정적이지
    # 않다 — 그래서 비교는 코드가 담당하고, LLM은 파싱까지만 맡긴다.
    신청일이후: Optional[str] = Field(
        None,
        description="질문이 '~이후/~부터/~이후에 등록된/신청된' 등으로 날짜 하한을 요구하면"
        "YYYY-MM-DD 형식으로 채워라(예: '2024년 1월 1일 이후' → '2024-01-01'). 신청이력에만"
        "의미가 있는 개념이다. 날짜 하한 조건이 없으면 null",
    )
    신청일이전: Optional[str] = Field(
        None,
        description="질문이 '~이전/~까지/~이전에 등록된/신청된' 등으로 날짜 상한을 요구하면"
        "YYYY-MM-DD 형식으로 채워라(예: '2024년 12월 31일까지' → '2024-12-31'). 신청이력에만"
        "의미가 있는 개념이다. 날짜 상한 조건이 없으면 null",
    )


class QueryAnalysis(BaseModel):
    standalone_question: str = Field(description="이전 대화 맥락을 반영해 이 질문만으로 검색 가능하도록 다시 쓴 독립형 질문")
    filters: FilterExtraction
    is_aggregation_question: bool = Field(
        description="질문이 특정 건을 찾는 게 아니라 전체 데이터에 대한 집계·순위·통계"
        "(총 개수, 가장 많은/적은, 최댓값·최솟값, 평균, 합계 등)를 요구하거나, 아무 조건 없이"
        "전체 목록을 통째로 요구하면 true. 단 특정 항목(업체명·담당자·결재자 등)의 전체 목록"
        "자체를 묻는 질문은 is_full_list_question으로 별도 처리하니 여기 포함하지 마라"
    )
    is_full_list_question: bool = Field(
        description="질문이 조건 없이 특정 항목(업체명·담당자·결재자 등)의 전체 목록을 요청하면"
        "true(예: '등록된 업체 리스트 알려줘', '담당자 누구누구 있어?'). 이런 목록은 DB에서"
        "직접 정확하게 가져올 수 있는 예외 케이스라 is_aggregation_question과 별개로 처리한다"
    )
    list_field: Optional[Literal["업체명", "담당자", "결재자"]] = Field(
        None, description="is_full_list_question이 true일 때, 어떤 항목의 목록인지"
    )
    is_out_of_scope: bool = Field(
        False,
        description="질문이 이력관리시스템 자료(기관·이력·첨부 등)를 찾으려는 게 아니라"
        "인사말·잡담·챗봇 자체에 대한 질문(예: '안녕', '고마워', '너는 누구야', '오늘 날씨 어때')"
        "이거나, 이력관리시스템 업무와 완전히 무관한 일반 요청(예: '파이썬 코드 짜줘', '오늘"
        "뉴스 알려줘', '피자 레시피 알려줘', '세종대왕이 누구야', '1+1은 얼마야')이면 true."
        "검색 대상이 애매하거나 짧아도(예: '국가', '수수료') 실제로 이 시스템 자료를 찾으려는"
        "의도가 있으면 false로 둬라 — 정말 검색 의도가 전혀 없을 때만 true로 한다",
    )


class AnswerGroup(BaseModel):
    source_index: int = Field(description="이 그룹이 설명하는 [자료 N]의 번호(1-based). 이 그룹은 오직 이 자료 하나만 다룬다")
    narrative: str = Field(
        description="source_index가 가리키는 자료 하나에 대한 설명. 다른 자료 내용을 섞지 마라. "
        "항목 나열이 아니라 자연스러운 문장 몇 개로 서술"
    )


class GroundedAnswer(BaseModel):
    no_result: bool = Field(False, description="[자료]에서 질문과 관련된 근거를 전혀 찾지 못했으면 true")
    summary: str = Field(
        description="항상 채움. 좁은 질문이면 이 필드 하나가 전체 답변 문장. "
        "넓은 질문(여러 건)이면 핵심 요약 한 문장만 담고 각 건의 자세한 설명은 groups에 넣음"
    )
    groups: list[AnswerGroup] = Field(default_factory=list, description="여러 건을 각각 설명해야 할 때만 채움")
    used_indices: list[int] = Field(default_factory=list, description="답변 근거로 실제 사용한 자료 번호(1-based)")
