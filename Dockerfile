FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# 패키지 설치
# combined_ca_bundle.pem: 이 개발 PC의 사내망 TLS 인터셉션(MonitorApp) 대응 — 빌드 시점엔
# pip install이 PyPI를 신뢰하는 데 쓰이고, 런타임엔 .env의 SSL_CERT_FILE/REQUESTS_CA_BUNDLE이
# 같은 파일을 가리켜 OpenAI 호출에 쓰므로 지우지 않고 이미지에 남겨둔다(274KB, 무해함).
# 운영에서 동일한 TLS 인터셉션이 없다면 운영용 .env에서 이 두 변수를 비워두면 되고,
# 그래도 파일 자체는 남아있어도 전혀 문제되지 않는다.
COPY requirements.txt combined_ca_bundle.pem ./
RUN PIP_CERT=/app/combined_ca_bundle.pem pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY .env .

# tiktoken 인코딩 파일 미리 받아두기 — tiktoken은 처음 쓰일 때 OpenAI API가 아니라
# openaipublic.blob.core.windows.net(완전히 다른 도메인)에서 자체적으로 사전 파일을
# 받아오는데, 운영 폐쇄망은 OpenAI API 경로만 열려있어 그 도메인은 막힌다
# (2026-10-29, attachment_auto_ingest 최초 실행 시 ConnectTimeoutError로 발견).
# 이 PC(인터넷 가능)에서 미리 받아 이미지에 넣어두고, TIKTOKEN_CACHE_DIR로 런타임에도
# 그 캐시를 그대로 쓰도록 고정해 다시 다운로드를 시도하지 않게 한다.
ENV TIKTOKEN_CACHE_DIR=/app/tiktoken_cache
RUN REQUESTS_CA_BUNDLE=/app/combined_ca_bundle.pem python -c \
    "import tiktoken; tiktoken.encoding_for_model('text-embedding-3-large')"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
