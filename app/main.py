from fastapi import FastAPI

from app.api import router

app = FastAPI(title="교부기관 이력관리시스템 챗봇")
app.include_router(router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
