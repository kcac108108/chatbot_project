"""2026-11-05 — 백필된 모든 첨부(파일경로)에 대해, 로컬 ATTACHMENT_BASE_DIR 기준으로
실제 파일이 존재하는지 전수 확인한다. /attachment-download 엔드포인트가 쓰는 것과
동일한 정규식/경로 조립 로직을 그대로 재사용한다(app/api/chat.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.chat import _FILE_PATH_RE  # 엔드포인트와 완전히 동일한 정규식을 그대로 재사용
from app.core.config import ATTACHMENT_BASE_DIR, QDRANT_COLLECTION
from app.core.vectordb import get_client


def run():
    client = get_client()
    total = 0
    ok = 0
    missing: list[tuple] = []
    bad_pattern: list[str] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION, limit=200, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for p in points:
            fp = p.payload.get("파일경로")
            if not fp:
                continue
            total += 1
            m = _FILE_PATH_RE.match(fp)
            if not m:
                bad_pattern.append(fp)
                continue
            docucd, attfile_id, filename = m.groups()
            local_path = Path(ATTACHMENT_BASE_DIR) / docucd / attfile_id / filename
            if local_path.exists():
                ok += 1
            else:
                missing.append((p.payload.get("데이터유형"), p.payload.get("원본ID"), fp, str(local_path)))
        if offset is None:
            break

    print(f"ATTACHMENT_BASE_DIR = {ATTACHMENT_BASE_DIR}")
    print(f"파일경로 있는 청크 {total}건 중: 로컬 파일 존재 {ok}건, 패턴불일치 {len(bad_pattern)}건, 로컬에 없음 {len(missing)}건")
    if bad_pattern:
        print("\n패턴 불일치 샘플:")
        for fp in bad_pattern[:10]:
            print(" ", fp)
    if missing:
        print(f"\n로컬에 없는 건 (최대 20개 표시):")
        for data_type, source_id, fp, local in missing[:20]:
            print(f"  [{data_type}] {source_id}  저장값={fp}  로컬시도경로={local}")


if __name__ == "__main__":
    run()
