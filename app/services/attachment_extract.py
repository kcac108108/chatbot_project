"""엑셀·한글·텍스트 등 원래부터 디지털 텍스트인 첨부파일에서 비전모델 없이 직접 텍스트를
추출한다. 스캔본 PDF·이미지처럼 "그림 속 글자"가 아니라 파일 안에 이미 구조화된 텍스트로
저장돼 있는 형식이라, OCR·비전모델보다 정확하고 비용도 들지 않는다.

스캔본 PDF·이미지·zip은 이 파일 하단의 비전모델 경로(`extract_any`)가 처리한다."""
import base64
import io
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import lxml.html
import olefile
import openpyxl
import pypdfium2 as pdfium
import xlrd

from app.core.openai_client import vision_extract_text


def _find_hwp5html() -> str:
    # pip이 설치하는 콘솔 스크립트는 현재 파이썬 실행파일과 같은 디렉터리(venv의
    # Scripts/ 또는 bin/)에 놓이는데, 그 디렉터리가 PATH에 없으면 shutil.which로는
    # 못 찾는다 — sys.executable 기준으로 직접 찾는 게 더 확실하다(윈도우/리눅스 둘 다 대응).
    exe_dir = Path(sys.executable).parent
    for candidate in (exe_dir / "hwp5html.exe", exe_dir / "hwp5html"):
        if candidate.exists():
            return str(candidate)
    found = shutil.which("hwp5html")
    if found:
        return found
    raise FileNotFoundError("hwp5html 실행파일을 찾을 수 없습니다 (pyhwp 설치 확인 필요)")


HWP5HTML_CMD = _find_hwp5html()


def _clean_cell(text: str) -> str:
    # 셀 하나 안에 줄바꿈(문단 여러 개)이 들어있으면 공백으로 합친다 — 표 한 행이 한 줄로
    # 안 나오고 여러 줄로 쪼개져 보이는 문제 방지(실제 한글 파일에서 재현됨).
    return " ".join(text.split())


def extract_hwp(path: Path) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([HWP5HTML_CMD, str(path), "--output", tmp], check=True, capture_output=True)
        tree = lxml.html.parse(str(Path(tmp) / "index.xhtml"))
        lines: list[str] = []
        for table in tree.xpath("//table"):
            for tr in table.xpath(".//tr"):
                cells = [_clean_cell(c.text_content()) for c in tr.xpath(".//td")]
                cells = [c for c in cells if c]
                if cells:
                    lines.append(" | ".join(cells))
            table.getparent().remove(table)
        for p in tree.xpath("//p"):
            text = _clean_cell(p.text_content())
            if text:
                lines.append(text)
        return "\n".join(lines)


def extract_xlsx(path: Path) -> str:
    wb = openpyxl.load_workbook(path, data_only=True)
    lines: list[str] = []
    for sheet in wb.worksheets:
        lines.append(f"[시트: {sheet.title}]")
        for row in sheet.iter_rows(values_only=True):
            cells = [_clean_cell(str(c)) for c in row if c is not None]
            cells = [c for c in cells if c]
            if cells:
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def extract_xls(path: Path) -> str:
    wb = xlrd.open_workbook(str(path))
    lines: list[str] = []
    for sheet in wb.sheets():
        lines.append(f"[시트: {sheet.name}]")
        for r in range(sheet.nrows):
            cells = [_clean_cell(str(c)) for c in sheet.row_values(r) if c not in (None, "")]
            cells = [c for c in cells if c]
            if cells:
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def extract_text_file(path: Path) -> str:
    # .sql/.txt는 옛 한국 시스템에서 CP949(EUC-KR)로 저장된 경우가 실제로 있음 — UTF-8로
    # 강제 읽으면 예외 없이 조용히 한글 주석이 깨져 나오는 문제가 재현됨(errors="replace"가
    # 오류를 삼켜버려서 알아채기 어려움). UTF-8을 먼저 시도하고 안 되면 CP949로 재시도한다.
    raw = path.read_bytes()
    for encoding in ("utf-8", "cp949"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def extract_doc(path: Path) -> str:
    """옛 워드(.doc, OLE2 바이너리 포맷) 파일에서 텍스트를 추출한다. python-docx는 .docx만
    지원해서 못 쓰고, 정식 바이너리 포맷 파싱(FIB·피스 테이블)은 복잡해서, WordDocument
    스트림을 UTF-16LE로 읽어 한글·영문 텍스트로 보이는 구간만 정규식으로 뽑아내는 실용적인
    방식을 쓴다 — 서식 정보는 유실되지만 본문 텍스트는 실제 파일로 검증 완료."""
    ole = olefile.OleFileIO(str(path))
    try:
        data = ole.openstream("WordDocument").read()
    finally:
        ole.close()
    text = data.decode("utf-16-le", errors="ignore")
    runs = re.findall(r"[가-힣 -~]{4,}", text)
    return "\n".join(runs)


DIRECT_EXTRACTORS = {
    ".hwp": extract_hwp,
    ".xlsx": extract_xlsx,
    ".xls": extract_xls,
    ".txt": extract_text_file,
    ".sql": extract_text_file,
    ".doc": extract_doc,
}


def extract_direct(path: Path) -> str | None:
    """비전모델이 필요 없는 형식이면 텍스트를 추출해 반환, 지원 안 하는 형식이면 None."""
    extractor = DIRECT_EXTRACTORS.get(path.suffix.lower())
    return extractor(path) if extractor else None


# 실제 추출 결과 검토 중 "수입"↔"수익", "순중량"↔"손중량"처럼 모양이 비슷한 한글을
# 헷갈리는 사례가 발견됨 — 해상도를 2.0(≈144dpi)에서 3.0(≈216dpi)으로 올려 글자를
# 더 선명하게 보여줘서 완화를 시도(비용·이미지 크기는 늘어남).
_PDF_RENDER_SCALE = 3.0


def _page_to_base64_png(page: "pdfium.PdfPage") -> str:
    bitmap = page.render(scale=_PDF_RENDER_SCALE)
    pil_image = bitmap.to_pil()
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_pdf_vision(path: Path) -> str:
    """스캔본 PDF는 페이지마다 이미지로 렌더링해 비전모델에 보내 텍스트를 옮겨 적게 한다."""
    pdf = pdfium.PdfDocument(str(path))
    try:
        pages_text = []
        for i, page in enumerate(pdf):
            b64 = _page_to_base64_png(page)
            text = vision_extract_text(b64).strip()
            if text:
                pages_text.append(f"[페이지 {i + 1}]\n{text}")
        return "\n\n".join(pages_text)
    finally:
        # Windows에서 pdfium이 파일 핸들을 계속 쥐고 있으면, 이 PDF가 zip 안에서
        # 임시폴더로 풀린 경우 TemporaryDirectory 정리 시 PermissionError가 남 — 명시적으로 닫아준다.
        pdf.close()


def extract_image_vision(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return vision_extract_text(b64).strip()


VISION_EXTRACTORS = {
    ".pdf": extract_pdf_vision,
    ".png": extract_image_vision,
    ".jpg": extract_image_vision,
    ".jpeg": extract_image_vision,
}


def _fix_zip_name(name: str) -> str:
    # 옛날 방식(UTF-8 플래그 없이 CP949)으로 만들어진 zip은 파이썬 zipfile이 이름을
    # CP437로 잘못 디코딩해 한글이 깨져 나옴(실제로 재현됨) — 반대로 UTF-8 플래그가 있는
    # zip(예: myCTI)은 이미 정상 디코딩됨. 이미 정상인 한글은 CP437에 없는 문자라
    # encode 단계에서 예외가 나므로, 그 경우 원본을 그대로 두면 안전하게 구분된다.
    try:
        return name.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def extract_zip(path: Path) -> str:
    """zip 자체가 TM_ATTFILE에 이미 첨부 하나로 등록돼 있으므로, 내부 파일을 DB와 새로
    매칭할 필요 없이 zip이 물려받은 식별정보를 그대로 쓴다 — 내부 파일 각각을 형식에 맞게
    (직접추출 또는 비전) 추출해 파일명 구분과 함께 이어붙인다."""
    parts = []
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(path) as z:
            for name in sorted(z.namelist()):
                if name.endswith("/"):
                    continue
                display_name = _fix_zip_name(name)
                extracted_path = Path(z.extract(name, tmp))
                text = (extract_any(extracted_path) or "").strip()
                if text:
                    parts.append(f"[{display_name}]\n{text}")
    return "\n\n".join(parts)


def extract_any(path: Path) -> str | None:
    """모든 형식을 아우르는 통합 추출 진입점 — 직접추출 가능하면 그걸로, 스캔본
    PDF·이미지면 비전모델로, zip이면 내부 파일들을 재귀적으로 처리한다."""
    suffix = path.suffix.lower()
    if suffix == ".zip":
        return extract_zip(path)
    direct = extract_direct(path)
    if direct is not None:
        return direct
    vision_extractor = VISION_EXTRACTORS.get(suffix)
    return vision_extractor(path) if vision_extractor else None
