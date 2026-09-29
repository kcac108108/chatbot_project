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

# /chat 요청 전체의 제한시간 — OpenAI 호출이 느려지거나(스키마 최초 컴파일, 큰 컨텍스트)
# 드물게 멈추는 경우, 넘으면 강제로 포기하고 안전한 안내 문구를 반환한다(2026-09-28 발견,
# 실제 1시간 넘게 응답이 안 오는 요청을 관찰함 — 운영 배포 전 반드시 필요한 방어).
CHAT_REQUEST_TIMEOUT_SECONDS = int(os.getenv("CHAT_REQUEST_TIMEOUT_SECONDS", "45"))

# 신규 이력 첨부파일 자동처리(2026-09-29) — 전산팀 OCR 서버(개발기간 외부망/운영 내부망
# 경로가 다름, ATTACHMENT_BASE_DIR과 동일한 전환 방식)와의 SFTP 접속정보. 포트·계정은
# 전산팀이 SFTP 호환성 조사 중이라 아직 확정 전 — 값이 없으면 자동처리 파이프라인이
# 실행 시점에 명확한 에러로 즉시 알려주고(조용히 실패하지 않고), 나머지 로직은 이 값들과
# 무관하게 먼저 완성해둔다.
OCR_SFTP_HOST = os.getenv("OCR_SFTP_HOST")
OCR_SFTP_PORT = int(os.getenv("OCR_SFTP_PORT", "22"))
OCR_SFTP_USER = os.getenv("OCR_SFTP_USER")
OCR_SFTP_PASSWORD = os.getenv("OCR_SFTP_PASSWORD")

# OCR 트리거 API — 개발 기간 한정 외부망 주소를 기본값으로 두고, 운영 배포 시 내부망
# 주소(http://193.1.2.43:9091)로 .env만 교체하면 된다.
OCR_API_BASE_URL = os.getenv("OCR_API_BASE_URL", "https://ai.bandtrass.or.kr:9091")

# OCR 서버(193.1.2.43)의 입력·출력 폴더 경로 — 전산팀 안내 그대로.
OCR_REMOTE_PDF_DIR = os.getenv("OCR_REMOTE_PDF_DIR", "/DATA/backend/rag/data")
OCR_REMOTE_IMAGE_DIR = os.getenv("OCR_REMOTE_IMAGE_DIR", "/DATA/backend/rag/data/images")
# 이미지 결과 txt가 pdfToText와 같은 폴더에 나오는지는 전산팀에 명시적으로 확인받지
# 못했다 — 일단 PDF와 동일 폴더로 가정, 실제 연동 시 재확인 필요.
OCR_REMOTE_TEXT_DIR = os.getenv("OCR_REMOTE_TEXT_DIR", "/DATA/backend/rag/data/pdfToText")
