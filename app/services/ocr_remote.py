"""전산팀 OCR 서버 연동 — 스캔본 PDF·이미지를 OCR 입력 경로에 두고, 동기 API를
호출해 처리 완료까지 기다린 뒤, 결과 텍스트를 읽어온다.

2026-10-01 설계 확정: 당초(9/29) SFTP로 193.1.2.43에 파일을 업로드하는 방식을 시도했으나
32번 서버(첨부 원본)↔193.1.2.43(OCR 서버) 간 SFTP 버전 호환 문제로 막힘 — 이후 대화에서
"우리 운영서버(챗봇 앱 배포 위치)와 이 OCR 서버가 물리적으로 같은 서버(43번 서버)"라는
핵심 사실이 확인되어, SFTP/네트워크 전송 자체가 불필요해짐. 아래는 전부 **로컬 파일시스템
경로**를 다루는 코드로, 과거 SFTP 버전(paramiko 기반)을 완전히 대체한다.

핵심 사실(2026-09-29 전산팀 회신, 반드시 지킬 것 — SFTP 여부와 무관하게 그대로 적용됨):
- API 호출은 동기(synchronous)다 — 요청을 보내면 처리가 끝날 때까지 기다렸다가 결과
  JSON을 그대로 돌려준다. 폴링이나 별도 완료 신호는 필요 없다.
- 파일 1개당 처리 시간은 약 3초 — 파일 개수에 비례해 HTTP 타임아웃을 넉넉히 잡아야 한다.
- ⚠️ 동시 호출 절대 금지: "이전 처리가 안 끝난 상태에서 또 호출하면 파일이 덮어써진다"고
  명시적으로 확인받음. 이 모듈의 함수를 호출하는 자동화 코드는 반드시 한 번에 하나씩만
  실행되도록 보장해야 한다(이 모듈 자체는 그 직렬화를 보장하지 않음 — 호출부 책임).
- PDF와 이미지는 입력 폴더·트리거 API가 서로 다르다(PDF: batch2, 이미지: batch).
- 지금 수작업으로도 같은 폴더를 쓰고 있어 자동화와 시간대가 겹치면 위 "덮어쓰기" 위험이
  똑같이 발생한다 — 코드로 완전히 막을 수 없는 부분(운영 시간대 조율 필요).

파일명 충돌 방지: 같은 파일명이 서로 다른 첨부 건에 중복 등록된 사례가 실제 53건
확인된 바 있어, OCR 서버에 올릴 때는 원본 파일명이 아니라 호출부가 지정한 이름
(`건번호__순번__원본파일명` 규칙)을 써야 한다 — 그래서 `run_ocr_batch`는 (로컬 경로,
업로드용 이름) 쌍을 받는다.
"""
import shutil
from pathlib import Path

import httpx

from app.core.config import (
    OCR_API_BASE_URL,
    OCR_REMOTE_IMAGE_DIR,
    OCR_REMOTE_PDF_DIR,
    OCR_REMOTE_TEXT_DIR,
)

_PDF_API_PATH = "/api/ktspi-ai/ocr/batch2"
_IMAGE_API_PATH = "/api/ktspi-ai/ocr/batch"
_IMAGE_EXTS = (".png", ".jpg", ".jpeg")


class OCRConfigError(RuntimeError):
    """OCR 입력·출력 폴더가 아직 로컬에서 접근 가능하지 않은 상태에서 호출하면 여기서 막는다."""


class OCRProcessingError(RuntimeError):
    """겹쳐 호출 방지 등 OCR 처리 자체의 문제."""


def check_ocr_config() -> None:
    """자동처리 파이프라인 시작 시 가장 먼저 호출해서, OCR 폴더 접근 불가를 DB 조회
    등 다른 작업 전에 빨리 알려준다. 같은 서버 안의 로컬 경로이므로 존재 여부만 확인한다."""
    for path in (OCR_REMOTE_PDF_DIR, OCR_REMOTE_IMAGE_DIR, OCR_REMOTE_TEXT_DIR):
        if not Path(path).is_dir():
            raise OCRConfigError(
                f"OCR 폴더에 접근할 수 없습니다: {path} — 같은 서버에 배포됐는지, "
                ".env의 OCR_REMOTE_* 경로가 맞는지 확인하세요."
            )


def _remote_settings(upload_name: str) -> tuple[str, str]:
    """(입력 폴더, 트리거 API 경로) — 업로드용 이름의 확장자에 따라 PDF/이미지 경로가 다르다."""
    suffix = Path(upload_name).suffix.lower()
    if suffix == ".pdf":
        return OCR_REMOTE_PDF_DIR, _PDF_API_PATH
    if suffix in _IMAGE_EXTS:
        return OCR_REMOTE_IMAGE_DIR, _IMAGE_API_PATH
    raise ValueError(f"OCR 대상이 아닌 확장자: {suffix}")


def run_ocr_batch(
    items: list[tuple[Path, str]], timeout_per_file: float = 10.0
) -> tuple[dict[Path, str], dict[Path, str]]:
    """스캔본 PDF·이미지 파일들을 OCR 서버 입력 폴더로 복사하고 결과 텍스트를 받아온다.

    items는 (원본 로컬 경로, OCR 서버에 올릴 때 쓸 이름) 쌍의 목록이다 — 업로드용 이름은
    호출부가 `건번호__순번__원본파일명` 규칙으로 지어서 넘겨야 한다(파일명 중복 방지).
    반환값은 (성공, 실패) 튜플이며 키는 항상 **원본 로컬 경로**다 — 성공은
    {원본경로: 추출텍스트}, 실패는 {원본경로: 실패사유}. PDF와 이미지가 섞여 있으면
    각각 다른 폴더·API로 나눠 처리한다.

    ⚠️ 이 함수를 동시에 두 곳에서 호출하면 안 된다(전산팀 확인) — 호출부가 직렬화를
    보장해야 한다."""
    if not items:
        return {}, {}

    pdfs = [(p, n) for p, n in items if Path(n).suffix.lower() == ".pdf"]
    images = [(p, n) for p, n in items if Path(n).suffix.lower() in _IMAGE_EXTS]
    known = {p for p, _ in pdfs} | {p for p, _ in images}
    unknown = [p for p, _ in items if p not in known]
    if unknown:
        raise ValueError(f"OCR 대상이 아닌 파일이 섞여 있습니다: {unknown}")

    succeeded: dict[Path, str] = {}
    failed: dict[Path, str] = {}
    if pdfs:
        s, f = _run_ocr_group(pdfs, OCR_REMOTE_PDF_DIR, _PDF_API_PATH, timeout_per_file)
        succeeded.update(s)
        failed.update(f)
    if images:
        s, f = _run_ocr_group(images, OCR_REMOTE_IMAGE_DIR, _IMAGE_API_PATH, timeout_per_file)
        succeeded.update(s)
        failed.update(f)
    return succeeded, failed


def _run_ocr_group(
    items: list[tuple[Path, str]], remote_dir: str, api_path: str, timeout_per_file: float
) -> tuple[dict[Path, str], dict[Path, str]]:
    remote_dir_path = Path(remote_dir)

    # 겹쳐 호출 방지 안전장치 — 입력 폴더에 남은 파일이 있으면 이전 작업이 아직
    # 안 끝났거나 수작업 처리 중일 수 있어 중단한다(겹치면 파일이 덮어써짐, 전산팀 확인).
    existing = list(remote_dir_path.iterdir())
    if existing:
        raise OCRProcessingError(
            f"OCR 입력 폴더({remote_dir})가 비어있지 않습니다({len(existing)}개 파일) — "
            "이전 작업이 진행 중이거나 수작업과 겹칠 위험이 있어 자동화를 중단합니다."
        )

    name_map = {upload_name: src_path for src_path, upload_name in items}
    for src_path, upload_name in items:
        shutil.copy2(src_path, remote_dir_path / upload_name)

    # 파일 1개당 약 3초(전산팀 확인) — 여유를 두고 최소 60초는 보장.
    timeout = max(60.0, timeout_per_file * len(items))
    resp = httpx.post(
        f"{OCR_API_BASE_URL}{api_path}", headers={"accept": "application/json"}, timeout=timeout
    )
    resp.raise_for_status()
    data = resp.json() if resp.content else {"files": []}

    succeeded: dict[Path, str] = {}
    failed: dict[Path, str] = {}
    reported = set()
    text_dir = Path(OCR_REMOTE_TEXT_DIR)
    for file_info in data.get("files", []):
        # 응답 스키마 예시가 PDF 기준(pdfName)이라 이미지 응답도 같은 필드명을
        # 쓴다고 가정 — 실제 연동 시 다르면 이 부분만 고치면 된다.
        upload_name = file_info.get("pdfName")
        src_path = name_map.get(upload_name)
        if src_path is None:
            continue
        reported.add(upload_name)
        if not file_info.get("success", False):
            failed[src_path] = file_info.get("errorMessage") or "OCR 처리 실패(사유 미상)"
            continue
        txt_path = text_dir / f"{Path(upload_name).stem}.txt"
        try:
            succeeded[src_path] = txt_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            failed[src_path] = f"결과 파일을 찾지 못함: {txt_path}"
        finally:
            txt_path.unlink(missing_ok=True)

    # 응답 files[]에 아예 안 나온 파일 — 전산팀 확인상 실패 시 항목 자체가 안 나올
    # 수 있다고 했으니, 요청했는데 응답에 없는 파일도 실패로 취급한다.
    for upload_name, src_path in name_map.items():
        if upload_name not in reported and src_path not in failed:
            failed[src_path] = "OCR 응답에 결과가 포함되지 않음"

    # 입력 폴더에 복사했던 파일 정리 — 처리 후 자동으로 옮겨지지 않는 경우를 대비해
    # 방어적으로 삭제 시도(전산팀 확인: 삭제해도 무방, 원본은 별도 서버에 있음).
    for _, upload_name in items:
        (remote_dir_path / upload_name).unlink(missing_ok=True)

    return succeeded, failed
