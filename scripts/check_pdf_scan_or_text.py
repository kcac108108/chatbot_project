"""
첨부파일 PDF가 스캔본(이미지, OCR 필요)인지 텍스트 기반인지 판별하는 독립 스크립트.

이 프로젝트(chatbot_project)의 다른 코드에 의존하지 않는다 — 파일 하나만 복사해서
실데이터 접근 권한이 있는 PC(이 개발 PC 말고)에서 실행하면 된다.

사용법:
    pip install pypdf
    python check_pdf_scan_or_text.py <PDF 폴더 경로>

판별 기준: 각 PDF의 페이지에서 텍스트 추출을 시도해, 추출된 글자 수가 페이지당
평균 20자 미만이면 "스캔본(추정)"으로 분류한다. 텍스트 레이어가 있는 PDF는 보통
페이지당 수백~수천 자가 나오고, 이미지만 있는 스캔본은 0자에 가깝게 나온다.
"""

import sys
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    print("pypdf가 설치되어 있지 않습니다. 먼저 다음을 실행하세요: pip install pypdf")
    sys.exit(1)

TEXT_PER_PAGE_THRESHOLD = 20


def classify(pdf_path: Path) -> tuple[str, int, int]:
    reader = PdfReader(str(pdf_path))
    n_pages = len(reader.pages)
    total_chars = sum(len((page.extract_text() or "").strip()) for page in reader.pages)
    avg_chars = total_chars // n_pages if n_pages else 0
    label = "스캔본(추정)" if avg_chars < TEXT_PER_PAGE_THRESHOLD else "텍스트 기반"
    return label, n_pages, avg_chars


def main():
    if len(sys.argv) != 2:
        print(f"사용법: python {Path(sys.argv[0]).name} <PDF 폴더 경로>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"폴더를 찾을 수 없습니다: {folder}")
        sys.exit(1)

    pdf_files = sorted(folder.glob("*.pdf")) + sorted(folder.glob("*.PDF"))
    if not pdf_files:
        print(f"'{folder}'에서 PDF 파일을 찾지 못했습니다.")
        sys.exit(1)

    n_scanned = 0
    n_text = 0
    n_error = 0

    print(f"총 {len(pdf_files)}개 PDF 검사 시작\n")
    for path in pdf_files:
        try:
            label, n_pages, avg_chars = classify(path)
        except Exception as e:
            print(f"[오류] {path.name} — {e}")
            n_error += 1
            continue

        print(f"[{label}] {path.name} (페이지 {n_pages}, 페이지당 평균 {avg_chars}자)")
        if label == "스캔본(추정)":
            n_scanned += 1
        else:
            n_text += 1

    print("\n" + "=" * 50)
    print(f"결과: 텍스트 기반 {n_text}건 / 스캔본(추정) {n_scanned}건 / 오류 {n_error}건")
    if n_scanned:
        print("→ 스캔본이 하나라도 있으면 OCR 파이프라인(Tesseract 등)이 필요합니다.")
    if n_error:
        print("→ 오류난 파일은 암호화·손상 등 별도 확인이 필요할 수 있습니다.")


if __name__ == "__main__":
    main()
