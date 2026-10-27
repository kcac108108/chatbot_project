"""Qdrant 컬렉션 전체를 스냅샷 파일로 내보낸다 (운영 서버 이관용).

이 PC(개발)에서 실행 — Qdrant에 "지금 담긴 데이터 전체를 파일 하나로 떠내는" 기능(스냅샷)을
호출해 로컬 디스크에 저장한다. 결과 파일은 소스코드 압축본과는 별도로 운영서버에 전달하고,
그쪽에서는 scripts/qdrant_snapshot_restore.py로 복원한다.

실행: venv/Scripts/python.exe scripts/qdrant_snapshot_export.py
결과: qdrant_snapshot/<컬렉션명>-<타임스탬프>.snapshot
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from app.core.config import QDRANT_COLLECTION, QDRANT_HOST, QDRANT_PORT

BASE_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
OUT_DIR = Path(__file__).resolve().parents[1] / "qdrant_snapshot"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print(f"[1/2] 스냅샷 생성 요청 중... (컬렉션: {QDRANT_COLLECTION})")
    with httpx.Client(timeout=600.0) as client:
        resp = client.post(f"{BASE_URL}/collections/{QDRANT_COLLECTION}/snapshots")
        resp.raise_for_status()
        snapshot_name = resp.json()["result"]["name"]
        print(f"  생성됨: {snapshot_name}")

        print("[2/2] 스냅샷 파일 다운로드 중...")
        out_path = OUT_DIR / snapshot_name
        with client.stream(
            "GET", f"{BASE_URL}/collections/{QDRANT_COLLECTION}/snapshots/{snapshot_name}"
        ) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print()
    print(f"완료: {out_path} ({size_mb:.1f} MB)")
    print("이 파일을 소스코드 압축본과 별도로 운영서버(전산팀)에 전달하세요.")
    print("운영서버 쪽 복원은 scripts/qdrant_snapshot_restore.py 사용.")


if __name__ == "__main__":
    main()
