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

# 2단계: 첨부파일 실물 루트 경로 (DOCUCD별 하위폴더를 담은 디렉터리). 신규 이력
# 첨부파일 자동처리(전산팀 OCR 연동, 2026-10-12 설계 확정)도 이 경로를 그대로 쓴다 —
# 전산팀이 매일 새벽 배치로 원본과 같은 경로·파일명(확장자만 .txt)에 결과를 생성해주므로
# 별도 drop 경로나 OCR API 직접 호출이 필요 없다(과거 SFTP 기반 설계는 폐기됨).
ATTACHMENT_BASE_DIR = os.getenv("ATTACHMENT_BASE_DIR")

# /chat 요청 전체의 제한시간 — OpenAI 호출이 느려지거나(스키마 최초 컴파일, 큰 컨텍스트)
# 드물게 멈추는 경우, 넘으면 강제로 포기하고 안전한 안내 문구를 반환한다(2026-09-28 발견,
# 실제 1시간 넘게 응답이 안 오는 요청을 관찰함 — 운영 배포 전 반드시 필요한 방어).
CHAT_REQUEST_TIMEOUT_SECONDS = int(os.getenv("CHAT_REQUEST_TIMEOUT_SECONDS", "45"))
