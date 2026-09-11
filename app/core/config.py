import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

# MonitorApp TLS 인터셉션 대응 — CLAUDE.md "개발 PC 네트워크 제약" 참고
# httpx는 SSL_CERT_FILE 환경변수를 자동으로 타지 않으므로, SSL_CERT_PATH로 노출해
# httpx.Client(verify=SSL_CERT_PATH)처럼 명시적으로 넘겨줘야 함 (app/core/openai_client.py 참고)
_ssl_cert_file = os.getenv("SSL_CERT_FILE")
SSL_CERT_PATH = str((BASE_DIR / _ssl_cert_file).resolve()) if _ssl_cert_file else None
if SSL_CERT_PATH:
    os.environ.setdefault("SSL_CERT_FILE", SSL_CERT_PATH)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", SSL_CERT_PATH)

ORACLE_USER = os.getenv("ORACLE_USER")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD")
ORACLE_DSN = os.getenv("ORACLE_DSN")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "history_chatbot")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# 정의서 6장: 벡터 검색 top 5~8건
CHAT_TOP_K = int(os.getenv("CHAT_TOP_K", "6"))

# 2단계: 첨부파일 실물 루트 경로 (DOCUCD별 하위폴더를 담은 디렉터리)
ATTACHMENT_BASE_DIR = os.getenv("ATTACHMENT_BASE_DIR")
