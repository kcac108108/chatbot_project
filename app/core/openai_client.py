import re
import time

import httpx
from openai import OpenAI, RateLimitError
from pydantic import BaseModel

from app.core.config import OPENAI_API_KEY, OPENAI_CHAT_MODEL, OPENAI_EMBEDDING_MODEL, SSL_CERT_PATH

# MonitorApp TLS 인터셉션 대응 — httpx는 SSL_CERT_FILE 환경변수를 자동으로 읽지 않아 명시 필요.
# http_client를 None으로 넘기면 openai SDK가 내부적으로 자체 기본 클라이언트를 만드는데,
# 이 과정이 openai==1.51.0 + httpx==0.28.1 조합에서 깨진다(httpx가 구버전 방식의 proxies
# 인자 지원을 없앴음 — 2026-10-29, SSL_CERT_FILE이 비어있는 운영 환경에서 처음 발견된 크래시).
# 항상 직접 만든 httpx.Client를 넘겨 그 깨진 내부 경로 자체를 타지 않도록 한다.
_http_client = httpx.Client(verify=SSL_CERT_PATH) if SSL_CERT_PATH else httpx.Client()
_client = OpenAI(api_key=OPENAI_API_KEY, http_client=_http_client)


_MAX_RETRIES = 8


def _with_retry(call):
    """분당 토큰 한도(TPM)는 여러 종류의 호출(비전 추출·요약·임베딩)이 같은 조직 한도를
    공유하기 때문에, 첨부파일 배치 작업 중 어느 호출에서든 실제로 재현됨 — 지수 백오프로
    재시도한다."""
    for attempt in range(_MAX_RETRIES):
        try:
            return call()
        except RateLimitError:
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(min(2**attempt, 60))


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = _with_retry(lambda: _client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=texts))
    return [item.embedding for item in response.data]


def chat_parse(messages: list[dict], response_format: type[BaseModel], temperature: float = 0.0) -> BaseModel:
    completion = _with_retry(
        lambda: _client.beta.chat.completions.parse(
            model=OPENAI_CHAT_MODEL,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
        )
    )
    return completion.choices[0].message.parsed


_VISION_EXTRACT_PROMPT = (
    "다음 이미지에 있는 텍스트를 그대로 옮겨 적는 단순 전사(transcription) 작업이다. 문서 내용에 대해 판단하거나 "
    "응답을 거부하지 말고, 보이는 글자를 위에서부터 순서대로 빠짐없이 받아 적기만 하면 된다.\n"
    "- 페이지 안에 표와 일반 문단(제목, 안내문, 목록 등)이 함께 있으면 둘 다 옮겨야 한다. 표 앞이나 뒤에 있는 일반 "
    "텍스트를 표에 집중하느라 건너뛰지 마라 — 표는 표대로, 그 앞뒤 문단은 문단대로 순서 그대로 전부 포함하라\n"
    "- 표가 있으면 행 단위로 옮기되, 열이 3개 이상이면 2개로 줄이지 말고 열 개수만큼 '|'로 구분하라"
    "(예: 열이 '항목명/사용용도/해당법령' 3개면 '항목명 | 사용용도 | 해당법령'처럼 3칸 다 채워라). 셀이 비어있으면 빈칸으로 두되 열 개수는 유지하라\n"
    "- 얼룩·접힌 자국 등으로 일부가 흐릿해도, 나머지 읽을 수 있는 텍스트는 전부 옮겨 적어라. 일부가 안 보인다는 이유로 "
    "페이지 전체를 비어있다고 하거나 옮겨 적기를 거부하지 마라 — 정말로 텍스트가 하나도 없는 빈 페이지일 때만 빈 문자열을 반환하라\n"
    "- 도장·서명·로고 등 순수 이미지 요소는 무시하고 텍스트만 추출하라\n"
    "- 요약하거나 해석하지 말고 있는 그대로 옮겨 적기만 하라\n"
    "- 마크다운 코드블록(```)이나 다른 마크다운 서식으로 감싸지 말고 순수 텍스트로만 출력하라"
)


def _strip_code_fence(text: str) -> str:
    # 프롬프트로 "마크다운으로 감싸지 마라"고 지시해도 가끔 ```로 감싸서 응답하는 경우가
    # 실제로 재현됨 — 프롬프트만 믿지 말고 코드에서도 한 번 더 벗겨낸다.
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines).strip()
    return text


_REPETITION_RE = re.compile(r"(\S{1,10}\s?)\1{9,}")


def _collapse_repetition(text: str) -> str:
    # 표 안의 빈 칸/구분자(예: "/")를 옮기다가 LLM이 반복 루프에 빠져 같은 조각을
    # 수백 번씩 토해내는 현상이 실제로 재현됨(예: "/ / / / ..."가 수백 개) — 진짜 내용이
    # 아니라 쓰레기 텍스트라 청크·임베딩을 오염시키므로, 같은 조각이 10회 넘게 반복되면
    # 3회로 줄인다.
    return _REPETITION_RE.sub(lambda m: m.group(1) * 3, text)


_REFUSAL_RE = re.compile(r"죄송|처리할 수 없|옮겨 적을 수 없|전사할 수 없|응답을 거부")
_VISION_REFUSAL_RETRIES = 3


def _call_vision(image_b64: str) -> str:
    completion = _with_retry(
        lambda: _client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _VISION_EXTRACT_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                }
            ],
            temperature=0.0,
        )
    )
    text = _strip_code_fence(completion.choices[0].message.content or "")
    return _collapse_repetition(text)


def vision_extract_text(image_b64: str) -> str:
    """이미지(페이지 스캔본 등) 한 장에서 텍스트를 그대로 옮겨 적는다 — OCR과 달리 비전
    LLM이 표 구조를 이해하고 옮기므로, 라벨/값이 뒤섞이는 OCR 특유의 문제를 피할 수 있다.

    같은 이미지·같은 프롬프트(temperature=0)인데도 가끔 "죄송하지만 처리할 수 없습니다" 식으로
    거부하는 현상이 실제로 재현됨(다른 모델로 바꿔도 마찬가지, 특정 페이지 고정도 아님 —
    OpenAI API의 잔여 비결정성으로 추정) — 거부 문구가 감지되면 같은 이미지로 몇 번 더
    재시도해서, 그중 정상적으로 옮겨 적은 응답이 나오면 그걸 쓴다."""
    text = _call_vision(image_b64)
    for _ in range(_VISION_REFUSAL_RETRIES):
        if not _REFUSAL_RE.search(text):
            break
        text = _call_vision(image_b64)
    return text
