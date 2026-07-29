import httpx
from openai import OpenAI

from app.core.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, SSL_CERT_PATH

# MonitorApp TLS 인터셉션 대응 — httpx는 SSL_CERT_FILE 환경변수를 자동으로 읽지 않아 명시 필요
_http_client = httpx.Client(verify=SSL_CERT_PATH) if SSL_CERT_PATH else None
_client = OpenAI(api_key=OPENAI_API_KEY, http_client=_http_client)


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = _client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]
