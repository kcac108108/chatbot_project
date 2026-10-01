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

# 신규 이력 첨부파일 자동처리 — 전산팀 OCR 연동(2026-10-01 설계 확정).
# 당초(9/29) SFTP로 193.1.2.43에 파일을 올리는 방식을 시도했으나 32번 서버(첨부 원본)
# ↔193.1.2.43(OCR 서버) 간 SFTP 버전 호환 문제로 막힘 — 이후 대화에서 핵심 사실 확인:
# 우리 운영서버(챗봇 앱 배포 위치)와 이 OCR 서버가 물리적으로 같은 서버(43번 서버,
# 193.1.2.43)임. 그래서 SFTP/네트워크 전송 자체가 불필요해지고, 아래 경로들은 전부
# "로컬 파일시스템 경로"로 취급한다(ATTACHMENT_BASE_DIR과 같은 패턴) — OCR_SFTP_* 설정은
# 더 이상 필요 없어 제거함.

# OCR 트리거 API — 개발 기간 한정 외부망 주소를 기본값으로 두고, 운영 배포 시 내부망
# 주소(http://193.1.2.43:9091)로 .env만 교체하면 된다. 같은 서버에 배포되면
# localhost 주소로 바꿔도 된다(코드 변경 없이 .env만 교체).
OCR_API_BASE_URL = os.getenv("OCR_API_BASE_URL", "https://ai.bandtrass.or.kr:9091")

# OCR 서버(같은 43번 서버)의 입력·출력 폴더 — 전산팀 안내 그대로, 전부 로컬 경로.
OCR_REMOTE_PDF_DIR = os.getenv("OCR_REMOTE_PDF_DIR", "/DATA/backend/rag/data")
OCR_REMOTE_IMAGE_DIR = os.getenv("OCR_REMOTE_IMAGE_DIR", "/DATA/backend/rag/data/images")
# 이미지 결과 txt가 pdfToText와 같은 폴더에 나오는지는 전산팀에 명시적으로 확인받지
# 못했다 — 일단 PDF와 동일 폴더로 가정, 실제 연동 시 재확인 필요.
OCR_REMOTE_TEXT_DIR = os.getenv("OCR_REMOTE_TEXT_DIR", "/DATA/backend/rag/data/pdfToText")

# 전산팀이 신규 첨부파일을 매일 가져다주는 위치 — DOCUCD/ATTFILE_ID/파일명 폴더 구조를
# 그대로 유지해서 전달해주기로 함(2026-10-01 전산팀 수락). 정확한 경로는 아직 전산팀과
# 확정 전이라 임시값 — 확정되는 대로 .env에 반영할 것.
ATTACHMENT_DROP_DIR = os.getenv("ATTACHMENT_DROP_DIR", "/new_attachments")
