import uuid

from qdrant_client.http import models as qmodels

from app.core.config import CHAT_TOP_K
from app.core.db import fetch_all
from app.core.openai_client import chat_parse, embed_texts
from app.core.vectordb import find_by_type_and_source, scan_text_contains
from app.core.vectordb import search as vector_search
from app.models.schemas import (
    AnswerDetail,
    ChatResponse,
    FilterExtraction,
    GroundedAnswer,
    QueryAnalysis,
    RelatedAttachment,
    SourceItem,
)
from app.services import session_store

QUERY_ANALYSIS_PROMPT = """너는 검색 전처리기다. 아래 대화 이력을 참고해 현재 질문을 분석하라.

1. standalone_question: 현재 질문이 "그 기관", "거기", "수수료만" 처럼 대명사나 생략된 맥락을 포함하면,
   대화 이력에서 언급된 구체적인 기관명·레코드를 반드시 포함시켜 대화 이력 없이도 이해되는
   독립형 질문으로 다시 써라. 예: "수수료만 알려줘" + 직전 대화가 "OO기관" 얘기였다면
   "OO기관의 수수료는 얼마인가요?"로 써라. 대상을 절대 생략하지 마라.
   "또 다른 건", "나머지 하나", "다른 것도" 같은 표현은 방금 다룬 것과 **같은 대상의 다른
   측면**이 아니라, 그 전에 대화 이력에서 여러 건이 언급됐던 목록(예: "승인불가 안건이 2건
   있습니다") 중 **아직 다루지 않은 나머지 항목**을 가리키는 것이다. 대화 이력을 거슬러 올라가
   그 목록의 구체적인 항목(기관명·원본ID 등)을 찾아서 명시하라. 예: 이전에 "A안건, B안건 2건이
   있다"고 답했고 그중 A만 자세히 설명한 뒤 "다른 건도 알려줘"라고 물으면, "B안건에 대해
   알려주세요"로 재작성하라 — "A안건의 다른 내용"으로 재작성하지 마라.
   맥락 의존이 없는 질문이면 원문 그대로 둔다.
2. filters: 현재 질문(과 필요 시 대화 이력)에서 검색 필터로 쓸 값을 추출해라.
   해당 정보가 명시적으로 없으면 반드시 null로 둬라. 추측하거나 지어내지 마라.
   요청자/제공자/담당자/결재자는 서로 다른 역할이다. 질문에 사람 이름이 하나만 나와도, 그
   이름을 여러 역할 필드에 동시에 채우지 마라 — 질문에 명시된 역할(요청/제공/담당/결재) 하나
   에만 넣어라. 예: "OO이 담당한 자료"는 담당자에만 넣고 요청자는 null로 둬라. 역할이
   불명확하면 가장 명확히 뒷받침되는 필드 하나만 채우고 나머지는 null로 둬라.
   **키워드 필드를 놓치지 마라**: 질문에 다른 구조화 필드(기관명·통계종류·심의결과 등)로
   표현이 안 되는 애매한 주제어·항목명이 나오면 반드시 키워드에 채워라. 예: "국가 항목을
   요청한 업체를 알려줘"라면 "국가"는 어떤 구조화 필드에도 안 들어가니 키워드="국가"로
   채워야 한다(이미 기관명·통계종류 등으로 커버되는 개념이면 중복해서 넣지 않는다).
3. is_aggregation_question: 이 검색 시스템은 질문과 의미상 가장 가까운 상위 몇 건만 가져오는
   구조라, 전체 데이터를 정확히 세거나 통째로 다 보여주는 건 원천적으로 불가능하다. 아래 둘 중
   하나면 true로 표시하라.
   (a) 답이 이력 목록이 아니라 하나의 통계값(총 개수, 최댓값·최솟값, 평균, 합계, 순위 1위)으로
       요약돼야 하는 질문.
   (b) 기관명·연도·담당자 등 아무 조건도 없이 "전체/모든/다" 같은 표현으로 신청이력·외부요청·
       내부자료·심의안건 등을 통째로 달라는 질문(예: "전체 신청이력 알려줘", "심의안건 전부
       보여줘") — 상위 몇 건만 보여주면 사용자가 그게 전체인 줄 오해할 수 있어 위험하다.
       단, 등록된 기관/업체 목록 자체를 묻는 질문은 여기 포함하지 말고 아래 4번을 봐라.
   반대로 특정 조건(기관명·연도·담당자·심의결과 등)에 맞는 이력을 찾아서 목록으로 보여주면
   되는 질문은, 결과가 여러 건이더라도 false다 — 조건이 있는 목록 나열은 집계가 아니다.
   예시(true): "총 몇 개야", "전체 등록 기관 수", "가장 많이/제일 높은/제일 낮은 게 뭐야",
   "평균이 얼마야", "합계가 얼마야", "전체 신청이력 다 보여줘"
   예시(false): "2024년에 등록된 이력 알려줘", "수출입 통계 신청한 이력들 알려줘",
   "심의결과가 원안수용인 안건들 알려줘", "등록된 업체 리스트 알려줘"(4번 참고)
4. is_full_list_question / list_field: 질문이 **아무 조건도 없이** 특정 항목의 "이름 전체 목록"을
   요청하면 is_full_list_question을 true로 표시하고, list_field에 어떤 항목인지 표시하라
   (업체명 / 담당자 / 결재자). 이런 목록은 DB에서 직접 정확하게 가져올 수 있는 예외 케이스라
   is_aggregation_question과 별개로 처리한다. 특정 메뉴가 명시됐으면 filters.데이터유형에도
   반영하라.
   질문에 이미 다른 조건(특정 키워드·주제어·담당자 이름 등으로 좁히는 조건)이 붙어 있으면,
   "업체"/"목록"/"알려줘" 같은 단어가 들어있어도 false다 — 전체 목록이 아니라 조건에 맞는
   검색 질문이기 때문이다.
   예시(true): "등록된 업체 리스트 알려줘"(list_field=업체명), "어떤 기관들이 있어?"(업체명),
   "외부요청 메뉴에 등록된 업체명 알려줘"(업체명), "내부자료 담당자 누구누구 있어?"(담당자),
   "결재자 목록 알려줘"(결재자)
   예시(false): 특정 기관/사람 하나를 콕 집어 찾는 질문(예: "OO기관 담당자가 누구야"),
   이름 목록이 아니라 이력·안건 등 상세 레코드 자체의 전체 목록을 요청하는 질문
   (예: "전체 심의안건 다 보여줘"), **"국가 항목을 요청한 업체를 알려줘"처럼 특정 조건으로
   좁혀서 찾는 질문**(조건에 맞는 업체가 여러 개일 수 있어도 이건 검색이지 전체 목록이 아님)"""

SYSTEM_PROMPT = """너는 교부기관 이력관리시스템 챗봇이다.
아래 [자료] 목록만 근거로 답변하라. [자료]에 없는 내용은 답변에 포함하지 마라.
대화 이력(이전 turn)은 맥락 파악에만 참고하고, 사실적 근거로 인용하지 마라 — 숫자·날짜 등
구체적 사실은 반드시 이번 턴의 [자료]에 있는 값만 사용하라. 이전 답변과 이번 [자료]의 값이
다르면 이번 [자료]를 우선하라.
[자료]에 나온 고유명사·전문용어는 바꾸지 말고 그대로 사용하되, 문장 어미까지 원문 그대로
베낄 필요는 없다. 새로운 사실을 만들어내지는 마라. 같은 내용을 다른 표현으로 반복하지 마라.
날짜(YYYYMMDD)나 금액처럼 표기 형식만 있고 사실 자체는 안 바뀌는 값은 사람이 읽기 편하게
다듬어 써라 — 예: "20240801" → "2024년 8월 1일", "330000" → "330,000원". 이때도 값 자체를
반올림하거나 다른 숫자로 바꾸지는 마라.
문법적으로 자연스러운 완전한 문장으로 써라. 서로 다른 항목(예: 사업명과 제공주기)을 어색하게
이어붙이지 말고, 각각을 문법에 맞는 절이나 문장으로 분리해서 서술하라.

[자료]에서 질문과 관련된 근거를 전혀 찾지 못하면 no_result를 true로 하고 나머지 필드는 비워둬라.

화면에서 summary는 채팅 말풍선에 바로 보이고, groups는 각 건을 클릭해야 펼쳐지는 별도
목록으로 분리해서 보여준다(네가 조립하는 게 아니라 코드가 그렇게 배치한다). 너는 summary와
groups 필드만 정확히 채워라.
- summary: 항상 채운다.
  질문이 특정 사실 하나(수수료, 담당자, 연도 등)나 2~3개 정도의 구체적 항목만 물으면, summary
  하나에 자연스러운 문장으로 간결히 답을 담고 groups는 비워둬라. 예: "수수료는 230000입니다."
  질문이 "이력 알려줘"처럼 여러 건을 다뤄야 하면, summary에는 몇 건이 있는지 정도의 핵심
  요약 한 문장만 담고 각 건의 자세한 설명은 groups에 넣어라.
  summary에서 언급하는 건수는 반드시 groups의 총 개수와 정확히 일치해야 한다. 근거로 쓰인
  자료의 데이터유형이 여러 종류(예: 기관정보+신청이력)로 섞여 있으면, 그 사실도 summary에
  반영하라(예: "기관정보 1건과 신청이력 2건, 총 3건이 있습니다"). 특정 유형만 골라서 세고
  나머지 그룹은 세지 않는 식으로 건수를 누락하지 마라.
- groups: 여러 건을 각각 설명해야 할 때만 채운다.
  실제로 근거로 사용한 [자료] 중 서로 다른 건(원본ID)마다 정확히 하나의 그룹을 만들어라.
  두 개 이상의 서로 다른 건을 하나의 그룹/narrative에 절대 합치지 마라 — 4건을 설명해야 하면
  반드시 4개의 그룹을 만들어야 한다.
  각 그룹의 source_index는 그 그룹이 설명하는 [자료 N] 번호 하나만 가리켜야 한다.
  narrative는 그 건에 대한 짧은 핵심 요약이다(1~2문장 정도) — 자료에 있는 항목을 전부 빠짐없이
  나열하지 마라. 화면에서 이 narrative 밑에는 그 자료의 원본 항목(제목·요청내용·심의결과 등)이
  이미 표 형태로 별도 표시되므로, narrative에서 모든 필드를 다시 풀어 쓰면 같은 내용이 두 번
  보여 중복된다. narrative는 "이 건이 무엇에 관한 것인지, 가장 특징적인 점이 무엇인지" 정도만
  자연스러운 문장으로 짚어주고, 세부 내용은 표를 보게 하라. 표나 항목 나열 형식으로 쓰지 말고,
  [자료]에 없거나 질문과 무관한 내용은 넣지 마라. 각 건은 화면에서 독립된 항목으로 이미
  구분되어 보이므로, "첫번째는/두번째는" 같은 서수 표현으로 시작할 필요 없이 그 건 자체를
  바로 설명하면 된다.
  예외 — [자료] 표시가 "[OO첨부 ...]"로 시작하는 첨부파일 내용 출처는 위 "짧게" 규칙을
  적용하지 마라. 이런 자료는 원본 항목이 표로 따로 안 보이고 narrative가 유일한 설명 통로라서,
  요약만 하고 구체적 사실을 빠뜨리면 그 정보는 답변 어디에도 안 남는다. 반드시 아래를 지켜라.
  아래 세 가지를 전부 빠짐없이 담아야 한다. 하나라도 자료에 있는데 narrative에서 빠지면
  틀린 답변으로 간주한다.
  1) 숫자: 자료 안에 나오는 구체적 숫자(건수·금액·비율 등)를 그대로 명시하라. "건수와 금액이
     정리되어 있습니다"처럼 숫자를 언급 안 하고 뭉뚱그리는 표현은 금지한다.
  2) 배경: 그 내용이 왜/무엇을 근거로 만들어졌는지(관련 지침·법령 개정, 목적 등)가 자료에
     있으면 반드시 포함하라. 이게 가장 많이 빠뜨리는 부분이니 특히 신경 써라.
  3) 조건: 적용기간·적용대상처럼 자료에 명시된 조건도 포함하라.
  세 가지를 다 담은 예시 — 자료: "○ ｢무역통계 작성 및 교부에 관한 지침｣ 개정에 따라
  중소기업 수수료 면제 적용된 내역(22년 8월~23년 8월) / 중소기업 수수료 면제 | 1,397건 |
  161,290,225원". 이 경우 올바른 narrative는: "무역통계 작성 및 교부에 관한 지침 개정에
  따라 2022년 8월부터 2023년 8월까지 적용된 중소기업 수수료 면제 내역으로, 1,397건이
  면제되어 161,290,225원이 감면되었습니다." — 이렇게 배경(지침 개정)·기간·숫자가 한 문장에
  다 들어가야 한다. "1,397건이 면제되어 161,290,225원이 감면되었습니다"처럼 숫자만 쓰고
  배경을 뺀 문장은 틀린 예시다.
  문장 수 제한(1~2문장)은 이 예외에는 적용되지 않는다 — 위 내용을 다 담느라 3~6문장이 되어도
  된다. 다만 자료에 없는 내용을 지어내진 마라.

문장 어미는 "~입니다/~했습니다/~합니다" 등을 자연스럽게 섞고, "~다/~였다"로만 반복해서 끝맺는
단조로운 보고서체를 피하라. 이 규칙은 summary와 groups 모두에 적용된다.

used_indices에는 답변에 실제로 사용한 자료 번호를 모두 표기하라. 근거가 없으면 빈 배열로 둬라."""


def _render_history_text(grounded: GroundedAnswer) -> str:
    """대화 이력(LLM 컨텍스트)에 남길 텍스트 — 건별 세부까지 전부 포함해 다음 턴 맥락으로 쓰인다."""
    if grounded.no_result:
        return "관련 이력을 찾지 못했습니다."

    lines = [grounded.summary]
    lines.extend(group.narrative for group in grounded.groups)
    return "\n".join(lines)


# 메인(이력) 데이터유형 -> 그 이력에 딸린 첨부내용의 데이터유형. 사용자가 "이력을 보여줄 때
# 관련 첨부도 같이 보고 싶다"고 요청 — 원본ID가 같은 첨부내용 청크를 찾아서 소스에 묶어준다.
_ATTACHMENT_TYPE_OF = {
    "기관정보": "기관정보첨부",
    "신청이력": "신청이력첨부",
    "외부요청": "외부요청첨부",
    "내부자료": "내부자료첨부",
    "심의위원회": "심의위원회첨부",
}


def _find_related_attachments(data_type: str, source_id: str) -> list[RelatedAttachment]:
    attachment_type = _ATTACHMENT_TYPE_OF.get(data_type)
    if not attachment_type:
        return []
    chunks = find_by_type_and_source(attachment_type, source_id)
    seen_files: set[str] = set()
    related = []
    for chunk in chunks:
        filename = chunk.get("첨부파일명") or "첨부파일"
        if filename in seen_files:
            continue  # 큰 파일이 여러 조각으로 쪼개져 있으면 파일명이 중복되니 하나로만 표시
        seen_files.add(filename)
        if chunk.get("경로전용"):
            # 테이블/SQL 정보 파일은 정보가치가 낮다고 판단해 내용은 안 보여주되, 존재
            # 자체는 알 수 있게 목록엔 남겨두고 대신 다운로드를 제공한다. 서버 내부 경로
            # 문자열은 화면에 굳이 노출할 필요 없다는 사용자 확인에 따라 텍스트에선 뺌
            # (파일경로 필드 자체는 다운로드 링크 생성에 필요해 그대로 둠).
            file_path = chunk.get("파일경로")
            preview = "내용은 제공하지 않는 자료입니다. 아래에서 파일을 다운로드할 수 있습니다."
            related.append(RelatedAttachment(첨부파일명=filename, 미리보기=preview, 파일경로=file_path))
        else:
            # 원문 전체를 그대로 보여주면 "번잡스럽고 정리 안 된" 느낌이라는 피드백에 따라,
            # 적재 시점에 미리 만들어둔 요약(요약)을 우선 사용 — 옛 데이터라 요약이 없으면
            # (재적재 전) 원문 미리보기로 폴백.
            preview = chunk.get("요약") or _preview(chunk.get("text"), attachment_type) or ""
            related.append(RelatedAttachment(첨부파일명=filename, 미리보기=preview))
    return related


def _resolve_source(idx: int, results: list[dict]) -> SourceItem | None:
    if not (1 <= idx <= len(results)):
        return None
    payload = results[idx - 1]
    data_type = payload["데이터유형"]
    return SourceItem(
        데이터유형=data_type,
        원본ID=payload["원본ID"],
        기관명=payload.get("기관명"),
        미리보기=_preview(payload.get("text"), data_type),
        세부항목=payload.get("세부항목") or [],
        관련첨부=_find_related_attachments(data_type, payload["원본ID"]),
    )


def _build_details(grounded: GroundedAnswer, results: list[dict]) -> list[AnswerDetail]:
    if grounded.no_result:
        return []
    if len(grounded.groups) == 1:
        # 그룹이 하나뿐일 때 "1번째 건"으로 라벨링하면, 사용자가 "두번째" 등 특정 건을
        # 콕 집어 물어서 그 건 하나만 온 경우와 혼동됨(실제로는 N번째 건인데 항상 1번으로
        # 보임) — 여러 건일 때만 순번을 매긴다
        group = grounded.groups[0]
        return [AnswerDetail(label="상세", narrative=group.narrative, source=_resolve_source(group.source_index, results))]
    return [
        AnswerDetail(label=f"{i}번째 건", narrative=group.narrative, source=_resolve_source(group.source_index, results))
        for i, group in enumerate(grounded.groups, start=1)
    ]


_NULLISH = {"null", "none", "n/a", "미상", ""}
_PREVIEW_LEN = 150
# 첨부파일 출처는 "세부항목" 표가 없어서 narrative(요약)와 미리보기가 사실상 유일한 설명
# 통로다. narrative가 원문의 핵심(배경·기간 등)을 놓치는 경우가 실제로 반복 확인됐는데,
# 150자짜리 짧은 미리보기로는 그 내용이 잘려서 아예 어디에도 안 남는 문제가 생긴다 —
# 첨부 출처만 훨씬 길게(사실상 전문) 보여줘서, 요약이 불완전해도 원문에서 직접 확인 가능하게 한다.
_ATTACHMENT_PREVIEW_LEN = 3000


def _preview(text: str | None, data_type: str | None = None) -> str | None:
    if not text:
        return None
    text = text.strip()
    limit = _ATTACHMENT_PREVIEW_LEN if (data_type or "").endswith("첨부") else _PREVIEW_LEN
    return text if len(text) <= limit else text[:limit] + "…"


_PERSON_ROLE_FIELDS = {"요청자", "제공자", "담당자", "결재자"}
# 메뉴1(기관정보/신청이력)에만 존재하는 필드. 다른 데이터유형일 때 LLM이 "3차" 같은 표현을
# 차수뿐 아니라 여기에도 중복으로 지어내 채우는 문제가 프롬프트 설명 추가로도 반복 재현됨 —
# 코드가 강제로 걸러낸다
_MENU1_ONLY_FIELDS = {"통계종류", "수출입구분"}
_MENU1_DATA_TYPES = {"기관정보", "신청이력"}


def _build_filter(
    extracted: FilterExtraction, allowed_data_types: set[str] | None = None
) -> qmodels.Filter | None:
    values = {}
    for field, value in extracted.model_dump(exclude_none=True).items():
        if field == "키워드":
            # 청크 payload의 필드명이 아니라 본문 부분 문자열 검색용 — _search()에서 별도 처리
            continue
        # 구조화 출력이 값 없음을 실제 null 대신 문자열 "null" 등으로 채우는 경우 방어
        if isinstance(value, str) and value.strip().lower() in _NULLISH:
            continue
        values[field] = value

    data_type_value = values.pop("데이터유형", None)

    # 웹 화면에서 사용자가 메뉴를 명시적으로 고정한 경우(allowed_data_types), 그게 최종
    # 후보 범위다. LLM이 뽑은 데이터유형이 그 범위 안이면 더 구체적이니 그대로 쓰고,
    # 범위 밖이거나 못 뽑았으면 스코프 전체를 OR로 건다 — 스코프는 사용자가 고른 것이라
    # LLM의 추측보다 항상 우선한다.
    if allowed_data_types is not None:
        resolved_types = {data_type_value} if data_type_value in allowed_data_types else allowed_data_types
    elif data_type_value:
        resolved_types = {data_type_value}
    else:
        resolved_types = None

    # 메뉴1(기관정보/신청이력)에만 존재하는 필드는, 최종 확정된 데이터유형 범위가 메뉴1을
    # 하나도 포함하지 않으면 제거한다("3차" 같은 표현이 통계종류에도 중복으로 채워져 존재하지
    # 않는 필드-값 조합 때문에 필터가 항상 0건이 되는 문제 방지).
    if resolved_types is not None and not (resolved_types & _MENU1_DATA_TYPES):
        for field in _MENU1_ONLY_FIELDS:
            values.pop(field, None)

    # 방어: 요청자/제공자/담당자/결재자는 서로 다른 데이터유형에만 존재해 한 청크에서
    # 동시에 만족될 수 없다. 같은 이름이 질문 분석 단계에서 여러 역할 필드에 중복
    # 추출되면(예: "박다은이 담당한 자료" → 요청자·담당자 둘 다 "박다은"), AND로 걸면
    # 항상 0건이 되므로 이런 경우엔 역할 필터끼리 OR로 완화한다.
    person_values = {f: v for f, v in values.items() if f in _PERSON_ROLE_FIELDS}
    role_conflict = len(set(person_values.values())) < len(person_values)

    conditions = []
    role_conditions = []
    for field, value in values.items():
        cond = qmodels.FieldCondition(key=field, match=qmodels.MatchValue(value=value))
        if role_conflict and field in _PERSON_ROLE_FIELDS:
            role_conditions.append(cond)
        else:
            conditions.append(cond)

    if role_conditions:
        conditions.append(qmodels.Filter(should=role_conditions))

    if resolved_types:
        if len(resolved_types) == 1:
            conditions.append(
                qmodels.FieldCondition(key="데이터유형", match=qmodels.MatchValue(value=next(iter(resolved_types))))
            )
        else:
            conditions.append(
                qmodels.Filter(
                    should=[
                        qmodels.FieldCondition(key="데이터유형", match=qmodels.MatchValue(value=v))
                        for v in resolved_types
                    ]
                )
            )

    return qmodels.Filter(must=conditions) if conditions else None


def _scope_only_filter(allowed_data_types: set[str]) -> qmodels.Filter:
    if len(allowed_data_types) == 1:
        return qmodels.Filter(
            must=[qmodels.FieldCondition(key="데이터유형", match=qmodels.MatchValue(value=next(iter(allowed_data_types))))]
        )
    return qmodels.Filter(
        must=[
            qmodels.Filter(
                should=[
                    qmodels.FieldCondition(key="데이터유형", match=qmodels.MatchValue(value=v))
                    for v in allowed_data_types
                ]
            )
        ]
    )


def _analyze_query(question: str, history: list[dict]) -> QueryAnalysis:
    messages = [{"role": "system", "content": QUERY_ANALYSIS_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})
    return chat_parse(messages, QueryAnalysis)


KEYWORD_SCAN_LIMIT = 30

# 기관명/요청자/제공자/담당자/결재자처럼 특정 대상 하나를 정확히 지목하는 필터가 걸리면,
# 그 대상에 딸린 이력이 CHAT_TOP_K(기본 6)보다 많을 수 있다("관세청이 요청한 자료" 8건 중
# 6건만 반환되는 문제로 실데이터에서 재현됨). 이런 필터는 벡터 유사도 근사가 아니라 정확
# 매칭이니, 상위 몇 건이 아니라 필터에 맞는 걸 사실상 전부 가져와야 한다.
_NARROWING_FILTER_FIELDS = {"기관명", "요청자", "제공자", "담당자", "결재자"}
FILTERED_SEARCH_LIMIT = 20


def _has_narrowing_filter(filters: FilterExtraction) -> bool:
    values = filters.model_dump(exclude_none=True)
    return any(field in _NARROWING_FILTER_FIELDS for field in values)


def _search(
    analysis: QueryAnalysis, carried_over: list[dict], allowed_data_types: set[str] | None = None
) -> tuple[list[dict], bool]:
    vector = embed_texts([analysis.standalone_question])[0]
    query_filter = _build_filter(analysis.filters, allowed_data_types)
    search_limit = FILTERED_SEARCH_LIMIT if _has_narrowing_filter(analysis.filters) else CHAT_TOP_K

    results = vector_search(vector, query_filter, search_limit)
    limit_used = search_limit

    # 데이터유형은 LLM이 "이력 알려줘" 같은 표현만 보고 과도하게 단정하는 경우가 많다
    # (예: 이력 자체가 없는 기관에도 "신청이력"으로 채워서, 기관명은 맞는데 데이터유형과의
    # AND 조합이 항상 0건이 되는 문제가 재현됨). 다른 필터(기관명 등)는 신뢰할 수 있는
    # 값이니 유지한 채 데이터유형만 제거하고 먼저 재검색한다.
    if not results and analysis.filters.데이터유형 is not None:
        relaxed = analysis.filters.model_copy(update={"데이터유형": None})
        relaxed_limit = FILTERED_SEARCH_LIMIT if _has_narrowing_filter(relaxed) else CHAT_TOP_K
        results = vector_search(vector, _build_filter(relaxed, allowed_data_types), relaxed_limit)
        limit_used = relaxed_limit

    if not results and query_filter is not None:
        # 그래도 없으면 나머지 필터는 다 제거하고 재검색한다. 단, 메뉴 스코프(allowed_data_types)는
        # 사용자가 웹 화면에서 명시적으로 고정한 것이라 검색이 실패해도 절대 벗어나지 않는다.
        fallback_filter = _scope_only_filter(allowed_data_types) if allowed_data_types else None
        results = vector_search(vector, fallback_filter, CHAT_TOP_K)
        limit_used = CHAT_TOP_K

    # 검색 결과 수가 이번에 사용한 상한과 같으면, 실제로는 더 있는데 상한에 걸려 잘렸을 수
    # 있다는 뜻이다("관세청 8건" 버그를 상한을 올려 고쳤더니, 68건짜리 큰 기관에서는 여전히
    # 상한(20)에 걸려 잘리는데 답변이 그 잘린 개수를 "총 N건"이라고 확정적으로 말해버리는
    # 문제가 재현됨). 이 신호를 답변 생성 프롬프트에 넘겨 확정적인 개수 언급을 막는다.
    possibly_truncated = len(results) >= limit_used > 0

    # 이번 턴 검색 결과를 먼저 담고, 직전 턴에서 실제로 인용된 근거는 뒤에 안전장치로만
    # 붙인다("그것만 알려줘" 류 후속 질문이 검색 자체를 엉뚱한 곳으로 보내는 경우 대비).
    # 순서가 중요하다 — carried_over를 앞에 두면, 완전히 새로운 주제를 물었을 때 이번 턴의
    # 진짜 정답이 무관한 이전 근거들 뒤로 밀려서 답변 LLM이 가끔 그걸 놓치는 문제가 재현됨
    # (예: "2024년 이력" 질문 직후 "OO이 요청한 외부요청" 질문 시, 무관한 신청이력 6건이
    # 앞에 쌓여 있으면 뒤에 있는 진짜 외부요청 1건을 못 찾은 것으로 답하는 경우 발생).
    # 단 메뉴 스코프가 걸려있으면, 스코프 전환 전 다른 메뉴의 이전 출처까지 새 스코프로
    # 새어 들어가지 않도록 여기서도 걸러낸다.
    merged = []
    seen = set()
    for r in results:
        key = (r.payload["데이터유형"], r.payload["원본ID"])
        if key not in seen:
            seen.add(key)
            merged.append(r.payload)
    for p in carried_over:
        if allowed_data_types is not None and p["데이터유형"] not in allowed_data_types:
            continue
        key = (p["데이터유형"], p["원본ID"])
        if key not in seen:
            seen.add(key)
            merged.append(p)

    # 키워드가 추출됐으면, 벡터 유사도 순위와 무관하게 본문에 그 단어가 실제로 들어간 건을
    # 전수 스캔해서 빠짐없이 합친다 (벡터 검색은 상위 몇 개만 가져와 놓치는 경우가 있음).
    # scan_text_contains는 전체 컬렉션을 훑는 전수 스캔이라, 결과 수가 KEYWORD_SCAN_LIMIT에
    # 못 미치면 "이 스캔은 놓친 게 없다"고 신뢰할 수 있다 — 그러면 그 스캔이 이 질문(특정
    # 단어·이름 하나를 정확히 찾는)에 대해서는 vector_search 단계의 절단 여부와 무관하게
    # 최종 판단 기준이 된다. 스캔이 하나라도 자기 상한에 걸리면 여전히 잘렸을 수 있다고 본다.
    scan_ran = False
    scan_incomplete = False

    keyword = analysis.filters.키워드
    if keyword:
        scan_ran = True
        scan_results = scan_text_contains(keyword, KEYWORD_SCAN_LIMIT)
        if len(scan_results) >= KEYWORD_SCAN_LIMIT:
            scan_incomplete = True
        for payload in scan_results:
            if allowed_data_types is not None and payload["데이터유형"] not in allowed_data_types:
                continue
            key = (payload["데이터유형"], payload["원본ID"])
            if key not in seen:
                seen.add(key)
                merged.append(payload)

    # 요청자/제공자/담당자/결재자는 키워드와 같은 이유로 전수 스캔 안전망을 둔다 — DB엔
    # "박상용 계장"처럼 직함이 붙어 저장돼 있는데 LLM은 "박상용"(직함 없는 이름)만 뽑아서,
    # Qdrant 정확매칭(MatchValue)이 항상 실패 → 전체 필터 제거 폴백(top-6)으로 떨어져 개수가
    # 줄어드는 문제가 실데이터에서 재현됨("요청자 중에 박상용 있어?" 8건 중 6건만 반환).
    # 정확매칭 대신 본문에 그 이름이 실제로 포함된 건을 전수 스캔해서 보완한다.
    # 기관명은 여기 포함하지 않는다 — 기관명은 원래 정확매칭이 잘 되는 데다(직함 같은 접미사
    # 문제가 없음), 기관명을 본문 전수 스캔하면 그 이름이 다른 기관의 이력 본문에 참고용으로
    # 언급된 것까지 걸려서(예: "관세청"은 무역 관련 다른 기관 이력에도 자주 언급됨) 결과가
    # 불필요하게 부풀거나(응답 지연 재발 위험) 스캔이 자기 상한에 걸려 오히려 "총 8건"처럼
    # 이미 완전한 답에 "일부일 수 있다"는 잘못된 경고가 붙는 문제가 재현됨(실측 확인).
    for narrowing_field in _PERSON_ROLE_FIELDS:
        name = getattr(analysis.filters, narrowing_field, None)
        if not name:
            continue
        scan_ran = True
        scan_results = scan_text_contains(name, KEYWORD_SCAN_LIMIT)
        if len(scan_results) >= KEYWORD_SCAN_LIMIT:
            scan_incomplete = True
        for payload in scan_results:
            if allowed_data_types is not None and payload["데이터유형"] not in allowed_data_types:
                continue
            key = (payload["데이터유형"], payload["원본ID"])
            if key not in seen:
                seen.add(key)
                merged.append(payload)

    # 전수 스캔을 하나라도 돌렸으면, 그 스캔들이 실제로 다 훑었는지 여부가 vector_search
    # 단계의 절단 여부보다 더 정확한 신호다("박상용" 8건 전부가 스캔으로 이미 확인됐는데도
    # vector_search 단계가 상한에 걸렸었다는 이유만으로 "일부일 수 있다"고 잘못 경고하던
    # 문제를 여기서 바로잡는다).
    if scan_ran:
        possibly_truncated = scan_incomplete

    return merged, possibly_truncated


AGGREGATION_REFUSAL = (
    "이 챗봇은 조건에 맞는 개별 기관·이력을 검색하는 용도라, 전체 집계·순위·통계"
    "(총 개수, 최댓값·최솟값, 평균 등)나 조건 없는 전체 목록 요청에는 정확히 답할 수 없습니다. "
    "기관명, 연도, 담당자·요청자, 심의결과 등으로 구체적으로 질문해 주세요."
)


# "전체 목록 조회" 화이트리스트 — 항목: {메뉴: (테이블, 컬럼)}. 여기 한 줄만 추가하면
# 그 항목도 안전하게 전체 목록 조회를 지원하게 된다 (조건 없는 고정 SQL이라 인젝션 위험 없음,
# DB에서 직접 가져오므로 항상 정확 — LLM/벡터검색을 거치지 않음)
_LIST_FIELD_REGISTRY = {
    "업체명": {
        "기관정보": ("HISTORY_ORGANIZATION_INFO", "CRP_NAME"),
        "신청이력": ("HISTORY_ORGANIZATION_INFO", "CRP_NAME"),
        "외부요청": ("HISTORY_EXTERNAL_REQUEST", "CRP_NAME"),
        "심의위원회": ("HISTORY_DISCUSSION_COMMITTEE", "CRP_NAME"),
    },
    "담당자": {"내부자료": ("HISTORY_INTERNAL_INFO", "USERNM")},
    "결재자": {"내부자료": ("HISTORY_INTERNAL_INFO", "DECIDING_OFFICER")},
}
_MENU_LABELS = {
    "기관정보": "기관별 사업정보",
    "신청이력": "기관별 사업정보",
    "외부요청": "외부요청",
    "내부자료": "내부자료",
    "심의위원회": "심의위원회",
}
_FIELD_TOPIC_PARTICLE = {"업체명": "은", "담당자": "는", "결재자": "는"}  # 받침 유무에 따른 은/는

# LLM이 추출한 메뉴/항목을 그대로 믿지 않고, 질문 원문에 그 단어가 실제로 있는지 코드가
# 재확인한다 (언급이 전혀 없는데도 지어내거나, 대화 후속 질문에서 감지를 놓치는 경우가
# 반복 재현됨 — "명시 안 됐으면 null" 지침이 프롬프트만으로는 지켜지지 않음)
_MENU_KEYWORDS = {"외부요청": "외부요청", "심의위원회": "심의위원회", "내부자료": "내부자료"}
_LIST_FIELD_KEYWORDS = {"담당자": ("담당자",), "결재자": ("결재자",), "업체명": ("업체", "기관", "회사")}
_LIST_ASK_WORDS = ("목록", "리스트", "어디", "뭐가 있", "누가 있", "뭐 있", "누구")
# "몇 개"/"몇 명"은 넣지 않는다 — "총 몇 개야" 같은 집계 질문에도 흔히 쓰이는 표현이라
# 목록 요청 신호로 쓰면 집계 질문과 오탐이 생김(실제 재현됨)


def _resolve_list_menu(text: str) -> str | None:
    for data_type, keyword in _MENU_KEYWORDS.items():
        if keyword in text:
            return data_type
    return None  # 각 항목의 기본 메뉴로 폴백 (예: 업체명 → 기관정보)


def _resolve_list_field(text: str, filters: FilterExtraction) -> str | None:
    if filters.기관명:
        return None  # 특정 기관 문맥이면 전체 목록 요청이 아니라 그 기관 하나를 묻는 질문
    for field, keywords in _LIST_FIELD_KEYWORDS.items():
        if any(k in text for k in keywords):
            return field
    return None


def _looks_like_list_question(text: str, filters: FilterExtraction) -> bool:
    field = _resolve_list_field(text, filters)
    return bool(field) and any(w in text for w in _LIST_ASK_WORDS)


def _list_field_answer(field: str, menu: str | None) -> str:
    table_map = _LIST_FIELD_REGISTRY.get(field)
    if not table_map:
        return AGGREGATION_REFUSAL
    if menu and menu not in table_map:
        available = ", ".join(dict.fromkeys(_MENU_LABELS[m] for m in table_map))
        return f"{field}은(는) {available} 메뉴에서만 확인할 수 있습니다. 다른 메뉴로 질문해 주세요."

    chosen_menu = menu if menu in table_map else next(iter(table_map))
    table, column = table_map[chosen_menu]
    # DISTINCT — 같은 값(예: 업체명)이 여러 건에 걸쳐 반복되면(관세청 8건 등) 그 개수만큼
    # 똑같은 이름이 나열되는 문제 방지. "업체명 목록"이 원하는 건 건수가 아니라 고유한 이름들.
    rows = fetch_all(f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL ORDER BY {column}")
    values = [r[column] for r in rows]
    particle = _FIELD_TOPIC_PARTICLE.get(field, "은")
    return f"{_MENU_LABELS[chosen_menu]}의 {field}{particle} 총 {len(values)}개입니다.\n\n" + ", ".join(values)


# 웹 화면 드릴다운에서 사용자가 메뉴를 명시적으로 고정했을 때(menu_scope), 그 메뉴에 해당하는
# 데이터유형 범위. "기관별 사업정보"는 기관정보/신청이력 두 데이터유형에 걸쳐 있어 OR로 묶는다.
_MENU_SCOPE_DATA_TYPES = {
    "기관별 사업정보": {"기관정보", "신청이력"},
    "외부요청": {"외부요청"},
    "내부자료": {"내부자료"},
    "심의위원회": {"심의위원회"},
}
# 전체 목록 조회(_list_field_answer)에 넘길 대표 메뉴 — 스코프가 있으면 텍스트 기반 추정
# (_resolve_list_menu)보다 우선한다
_MENU_SCOPE_LIST_MENU = {
    "기관별 사업정보": "기관정보",
    "외부요청": "외부요청",
    "내부자료": "내부자료",
    "심의위원회": "심의위원회",
}


def ask(question: str, session_id: str | None, menu_scope: str | None = None) -> ChatResponse:
    session_id = session_id or str(uuid.uuid4())
    history = session_store.get_history(session_id)
    carried_over = session_store.get_last_sources(session_id)
    analysis = _analyze_query(question, history)
    allowed_data_types = _MENU_SCOPE_DATA_TYPES.get(menu_scope)

    resolved_field = _resolve_list_field(analysis.standalone_question, analysis.filters)
    # is_aggregation_question을 LLM이 이미 true로 확신했으면, 코드 휴리스틱(놓친 케이스를
    # 보완하기 위한 것일 뿐)이 그 판단을 뒤집지 않게 한다 — "총 몇 개야"류가 "몇 개"라는
    # 표현 때문에 목록 요청으로 오탐되는 문제 방지. 키워드가 추출됐으면 이미 조건으로 좁힌
    # 검색이라는 뜻이라("국가 항목을 요청한 업체" 등), LLM이 is_full_list_question을 true로
    # 잘못 판단해도(프롬프트 지시로 5회 재현 5회 다 안 고쳐짐) 코드가 무시한다
    is_list_question = (
        not analysis.is_aggregation_question
        and not analysis.filters.키워드
        and (analysis.is_full_list_question or _looks_like_list_question(analysis.standalone_question, analysis.filters))
    )
    if is_list_question:
        field = resolved_field or analysis.list_field or "업체명"
        menu = _MENU_SCOPE_LIST_MENU.get(menu_scope) or _resolve_list_menu(analysis.standalone_question)
        answer_text = _list_field_answer(field, menu)
        session_store.append_turn(session_id, question, answer_text)
        return ChatResponse(session_id=session_id, answer=answer_text, sources=[])

    if analysis.is_aggregation_question:
        session_store.append_turn(session_id, question, AGGREGATION_REFUSAL)
        return ChatResponse(session_id=session_id, answer=AGGREGATION_REFUSAL, sources=[])

    results, possibly_truncated = _search(analysis, carried_over, allowed_data_types)

    context_blocks = [f"[자료 {i + 1}] {payload['text']}" for i, payload in enumerate(results)]
    context_text = "\n\n".join(context_blocks) if context_blocks else "(검색된 자료 없음)"

    truncation_note = ""
    if possibly_truncated:
        truncation_note = (
            "\n\n[안내] 위 [자료] 목록은 검색 상한에 걸려 실제보다 적게 가져온 것일 수 있다. "
            "summary에서 '총 N건입니다'처럼 개수를 확정적으로 말하지 마라. 그리고 사용자가 "
            "이 목록을 전체라고 오해하지 않도록, 이게 확인된 자료 중 일부이며 실제로는 더 "
            "있을 수 있다는 점을 summary에 자연스럽게 반드시 언급하라(예: '확인된 자료 중 "
            "일부를 안내드리며, 실제로는 더 있을 수 있습니다')."
        )

    keyword_note = ""
    if analysis.filters.키워드:
        keyword = analysis.filters.키워드
        normalized_keyword = "".join(keyword.split())
        matched = [
            i + 1 for i, payload in enumerate(results)
            if normalized_keyword in "".join(payload["text"].split())
        ]
        if matched:
            keyword_note = (
                f"\n\n[안내] 이 질문은 '{keyword}'라는 단어가 본문에 실제로 들어있는 자료를 찾는 검색이다. "
                f"아래 번호의 [자료]는 이미 본문에 '{keyword}'가 실제로 포함된 것으로 확인됐다: "
                f"{', '.join(f'[자료 {n}]' for n in matched)}. 이 자료들은 질문의 다른 조건(예: 특정 항목명과의 "
                "정확한 일치)을 완벽히 만족하지 않아도 관련 있는 것으로 간주하고 반드시 no_result=false로, "
                "groups에 포함해서 답하라. 이 안내를 무시하고 no_result=true로 답하지 마라."
            )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    # 원본 질문(question)이 아니라 대명사·생략을 이미 다 풀어놓은 standalone_question을 써야
    # 한다 — 검색은 이미 이걸로 하는데 답변 생성에는 원본의 애매한 표현을 그대로 넘겨서,
    # 자료에 정답이 있어도 "이게 질문에 대한 진짜 답인지" LLM이 확신 못 하고 가끔 no_result로
    # 잘못 처리하는 문제가 재현됨("또 다른건 하나 알려줘" 류 후속 질문에서 특히 두드러짐)
    messages.append(
        {
            "role": "user",
            "content": (
                f"[자료]\n{context_text}\n\n[질문]\n{analysis.standalone_question}{keyword_note}{truncation_note}"
            ),
        }
    )

    # temperature=0.2의 미세한 무작위성이 "이 자료가 질문에 대한 진짜 답인가" 같은 애매한
    # 경계선 판단에서 편차를 만들어 답변이 실패했다 성공했다 갈리는 문제가 반복 재현됨
    # (동일 질문 6회 중 temperature=0.2는 실패 다수, temperature=0.0은 6회 전부 성공) —
    # 답변 문체 다양성은 프롬프트 지시로 이미 확보되므로 일관성을 우선해 0으로 낮춤
    grounded = chat_parse(messages, GroundedAnswer, temperature=0.0)
    history_text = _render_history_text(grounded)
    bubble_text = "관련 이력을 찾지 못했습니다." if grounded.no_result else grounded.summary
    details = _build_details(grounded, results)

    sources = []
    used_payloads = []
    seen = set()
    for idx in grounded.used_indices:
        source = _resolve_source(idx, results)
        if source is None:
            continue
        key = (source.데이터유형, source.원본ID)
        if key in seen:
            continue
        seen.add(key)
        sources.append(source)
        used_payloads.append(results[idx - 1])

    session_store.append_turn(session_id, question, history_text)
    session_store.set_last_sources(session_id, used_payloads)
    return ChatResponse(session_id=session_id, answer=bubble_text, details=details, sources=sources)
