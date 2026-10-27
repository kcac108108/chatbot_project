"""Qdrant 스냅샷 파일로 컬렉션을 복원한다 (운영 서버 배포용 — 전산팀 실행).

사전조건: docker-compose up으로 qdrant, app(chatbot-app) 컨테이너가 떠 있어야 함.

실행 순서(운영서버에서):
  1. 전달받은 .snapshot 파일을 chatbot-app 컨테이너 안으로 복사
       docker cp <스냅샷파일경로> chatbot-app:/app/snapshot.tmp
  2. 컨테이너 안에서 이 스크립트 실행
       docker exec chatbot-app python scripts/qdrant_snapshot_restore.py /app/snapshot.tmp
  3. 완료 후 컨테이너 안 임시 파일 정리(선택)
       docker exec chatbot-app rm /app/snapshot.tmp

복원 대상 컬렉션이 이미 있으면 스냅샷 내용으로 덮어써지고, 없으면 새로 만들어진다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from app.core.config import QDRANT_COLLECTION, QDRANT_HOST, QDRANT_PORT

BASE_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"


def main() -> None:
    if len(sys.argv) != 2:
        print("사용법: python scripts/qdrant_snapshot_restore.py <스냅샷파일경로>")
        sys.exit(1)

    snapshot_path = Path(sys.argv[1])
    if not snapshot_path.is_file():
        print(f"파일을 찾을 수 없음: {snapshot_path}")
        sys.exit(1)

    print(f"[1/1] 스냅샷 업로드 및 복원 중... (컬렉션: {QDRANT_COLLECTION})")
    with httpx.Client(timeout=1200.0) as client:
        with open(snapshot_path, "rb") as f:
            resp = client.post(
                f"{BASE_URL}/collections/{QDRANT_COLLECTION}/snapshots/upload",
                files={"snapshot": (snapshot_path.name, f)},
            )
        resp.raise_for_status()

    print()
    print("복원 완료.")
    print("확인: python scripts/verify_ingest.py 로 건수 대조 가능")


if __name__ == "__main__":
    main()
