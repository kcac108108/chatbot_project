"""⚠️ (2026-09-29 당일 저녁 논의로 방향 변경 가능성 높음, 내일 전산팀과 협의 예정)
32번 서버(첨부 원본)↔193.1.2.43(OCR 서버) 간 SFTP 버전 호환성 문제로, 전산팀이 대안으로
"매일 오전 6시 전산팀이 전일 생성 첨부파일을 자체적으로 OCR 처리해 운영 서버의 특정
로컬 경로에 결과 .txt를 직접 떨궈주는" 방식을 제안함 — 확정되면 이 모듈의 SFTP
업로드+API 트리거 로직 전체가 불필요해지고, "로컬 경로에서 완성된 .txt 읽기"로 훨씬
단순해질 예정(파일명 규칙·정확한 경로·처리 범위는 내일 협의 후 확정). 아래 내용은 그
협의 전, "우리가 직접 SFTP로 업로드하고 API를 호출한다"는 원래 설계 기준으로 작성됨 —
당장 폐기하지 않고 남겨두되, 실제 사용 전 반드시 내일 결정된 방향과 다시 대조할 것.

전산팀 OCR 서버(193.1.2.43) 연동 — SFTP로 스캔본 PDF·이미지를 올리고, 동기 API를
호출해 처리 완료까지 기다린 뒤, 결과 텍스트를 SFTP로 내려받는다.

핵심 사실(2026-09-29 전산팀 회신, 반드시 지킬 것):
- API 호출은 동기(synchronous)다 — 요청을 보내면 처리가 끝날 때까지 기다렸다가 결과
  JSON을 그대로 돌려준다. 폴링이나 별도 완료 신호는 필요 없다.
- 파일 1개당 처리 시간은 약 3초 — 파일 개수에 비례해 HTTP 타임아웃을 넉넉히 잡아야 한다.
- ⚠️ 동시 호출 절대 금지: "이전 처리가 안 끝난 상태에서 또 호출하면 파일이 덮어써진다"고
  명시적으로 확인받음. 이 모듈의 함수를 호출하는 자동화 코드는 반드시 한 번에 하나씩만
  실행되도록 보장해야 한다(이 모듈 자체는 그 직렬화를 보장하지 않음 — 호출부 책임).
- PDF와 이미지는 업로드 폴더·트리거 API가 서로 다르다(PDF: batch2, 이미지: batch).
- 지금 수작업으로도 같은 폴더를 쓰고 있어 자동화와 시간대가 겹치면 위 "덮어쓰기" 위험이
  똑같이 발생한다 — 코드로 완전히 막을 수 없는 부분(운영 시간대 조율 필요).

SFTP 접속정보(포트·계정)는 전산팀이 호환성(SFTP 버전) 문제를 조사 중이라 아직 미확정
상태다(2026-09-29) — 값이 없으면 OCRConfigError로 명확하게 실패한다(조용히 넘어가지 않음).
"""
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import httpx
import paramiko

from app.core.config import (
    OCR_API_BASE_URL,
    OCR_REMOTE_IMAGE_DIR,
    OCR_REMOTE_PDF_DIR,
    OCR_REMOTE_TEXT_DIR,
    OCR_SFTP_HOST,
    OCR_SFTP_PASSWORD,
    OCR_SFTP_PORT,
    OCR_SFTP_USER,
)

_PDF_API_PATH = "/api/ktspi-ai/ocr/batch2"
_IMAGE_API_PATH = "/api/ktspi-ai/ocr/batch"
_IMAGE_EXTS = (".png", ".jpg", ".jpeg")


class OCRConfigError(RuntimeError):
    """SFTP 접속정보가 아직 채워지지 않은 상태에서 호출하면 여기서 막는다."""


class OCRProcessingError(RuntimeError):
    """겹쳐 호출 방지 등 OCR 처리 자체의 문제."""


def check_ocr_config() -> None:
    """자동처리 파이프라인 시작 시 가장 먼저 호출해서, SFTP 정보 누락을 DB 조회 등
    다른 작업 전에 빨리 알려준다."""
    if not (OCR_SFTP_HOST and OCR_SFTP_USER and OCR_SFTP_PASSWORD):
        raise OCRConfigError(
            "OCR_SFTP_HOST/OCR_SFTP_USER/OCR_SFTP_PASSWORD가 설정되지 않았습니다 — "
            "전산팀 SFTP 호환성 조사 결과가 나오면 .env에 반영한 뒤 다시 시도하세요."
        )


@contextmanager
def _sftp_session() -> Iterator[paramiko.SFTPClient]:
    check_ocr_config()
    transport = paramiko.Transport((OCR_SFTP_HOST, OCR_SFTP_PORT))
    try:
        transport.connect(username=OCR_SFTP_USER, password=OCR_SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        try:
            yield sftp
        finally:
            sftp.close()
    finally:
        transport.close()


def _remote_settings(path: Path) -> tuple[str, str]:
    """(업로드 폴더, 트리거 API 경로) — 확장자에 따라 PDF/이미지 경로가 다르다."""
    if path.suffix.lower() == ".pdf":
        return OCR_REMOTE_PDF_DIR, _PDF_API_PATH
    if path.suffix.lower() in _IMAGE_EXTS:
        return OCR_REMOTE_IMAGE_DIR, _IMAGE_API_PATH
    raise ValueError(f"OCR 대상이 아닌 확장자: {path.suffix}")


def run_ocr_batch(
    local_paths: list[Path], timeout_per_file: float = 10.0
) -> tuple[dict[Path, str], dict[Path, str]]:
    """스캔본 PDF·이미지 파일들을 OCR 서버에 올리고 결과 텍스트를 받아온다.

    반환값은 (성공, 실패) 튜플 — 성공은 {로컬경로: 추출텍스트}, 실패는
    {로컬경로: 실패사유}. PDF와 이미지가 섞여 있으면 각각 다른 폴더·API로 나눠 처리한다.

    ⚠️ 이 함수를 동시에 두 곳에서 호출하면 안 된다(전산팀 확인) — 호출부가 직렬화를
    보장해야 한다."""
    if not local_paths:
        return {}, {}

    pdfs = [p for p in local_paths if p.suffix.lower() == ".pdf"]
    images = [p for p in local_paths if p.suffix.lower() in _IMAGE_EXTS]
    known = set(pdfs) | set(images)
    unknown = [p for p in local_paths if p not in known]
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
    paths: list[Path], remote_dir: str, api_path: str, timeout_per_file: float
) -> tuple[dict[Path, str], dict[Path, str]]:
    with _sftp_session() as sftp:
        # 겹쳐 호출 방지 안전장치 — 업로드 폴더에 남은 파일이 있으면 이전 작업이 아직
        # 안 끝났거나 수작업 처리 중일 수 있어 중단한다(겹치면 파일이 덮어써짐, 전산팀 확인).
        existing = sftp.listdir(remote_dir)
        if existing:
            raise OCRProcessingError(
                f"OCR 입력 폴더({remote_dir})가 비어있지 않습니다({len(existing)}개 파일) — "
                "이전 작업이 진행 중이거나 수작업과 겹칠 위험이 있어 자동화를 중단합니다."
            )

        name_map = {path.name: path for path in paths}
        for path in paths:
            sftp.put(str(path), f"{remote_dir}/{path.name}")

        # 파일 1개당 약 3초(전산팀 확인) — 여유를 두고 최소 60초는 보장.
        timeout = max(60.0, timeout_per_file * len(paths))
        resp = httpx.post(
            f"{OCR_API_BASE_URL}{api_path}", headers={"accept": "application/json"}, timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json() if resp.content else {"files": []}

        succeeded: dict[Path, str] = {}
        failed: dict[Path, str] = {}
        reported = set()
        for file_info in data.get("files", []):
            # 응답 스키마 예시가 PDF 기준(pdfName)이라 이미지 응답도 같은 필드명을
            # 쓴다고 가정 — 실제 연동 시 다르면 이 부분만 고치면 된다.
            name = file_info.get("pdfName")
            local_path = name_map.get(name)
            if local_path is None:
                continue
            reported.add(name)
            if not file_info.get("success", False):
                failed[local_path] = file_info.get("errorMessage") or "OCR 처리 실패(사유 미상)"
                continue
            txt_remote = f"{OCR_REMOTE_TEXT_DIR}/{Path(name).stem}.txt"
            try:
                with sftp.open(txt_remote, "r") as f:
                    succeeded[local_path] = f.read().decode("utf-8", errors="replace")
            except FileNotFoundError:
                failed[local_path] = f"결과 파일을 찾지 못함: {txt_remote}"
            finally:
                try:
                    sftp.remove(txt_remote)
                except (FileNotFoundError, OSError):
                    pass

        # 응답 files[]에 아예 안 나온 파일 — 전산팀 확인상 실패 시 항목 자체가 안 나올
        # 수 있다고 했으니, 요청했는데 응답에 없는 파일도 실패로 취급한다.
        for name, local_path in name_map.items():
            if name not in reported and local_path not in failed:
                failed[local_path] = "OCR 응답에 결과가 포함되지 않음"

        # 업로드했던 원본 정리 — 처리 후 자동으로 옮겨지지 않는 경우를 대비해 방어적으로
        # 삭제 시도(전산팀 확인: 삭제해도 무방, 원본은 별도 서버에 있음).
        for path in paths:
            try:
                sftp.remove(f"{remote_dir}/{path.name}")
            except (FileNotFoundError, OSError):
                pass

        return succeeded, failed
