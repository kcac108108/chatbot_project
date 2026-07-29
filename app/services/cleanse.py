import re

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_DECOR_SYMBOL_RE = re.compile(r"[■★▶□☆▷►◆◇▲△▼▽]")
_DIVIDER_RE = re.compile(r"[-=]{3,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def cleanse(value: str | None) -> str | None:
    """정의서 5.0절 공통 클렌징 규칙: HTML 태그·장식기호·구분선 제거, 연속 줄바꿈/공백 정리, 앞뒤 공백 제거."""
    if value is None:
        return None
    text = _HTML_TAG_RE.sub("", value)
    text = _DECOR_SYMBOL_RE.sub("", text)
    text = _DIVIDER_RE.sub("", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = text.strip()
    return text if text else None
