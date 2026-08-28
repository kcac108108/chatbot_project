"""
PaddleOCR로 이미지/PDF에서 텍스트를 추출해보는 테스트 스크립트 (2단계 사전 검증용).

사용법:
    python scripts/ocr_test.py <이미지 또는 PDF 파일 경로>

이 개발 PC(MonitorApp SSL 인터셉션 환경)에서 PaddleOCR을 쓰려면 두 가지 우회 설정이
필요하다는 걸 확인했다 — 아래 두 환경변수를 스크립트 안에서 직접 설정해뒀다:

1. SSL_CERT_FILE / REQUESTS_CA_BUNDLE: PaddleOCR이 첫 실행 시 모델 가중치를 인터넷에서
   받아오는데, 이 PC의 사설 루트 인증서를 안 걸어주면 "No available model hosting
   platforms detected" 에러로 실패한다(pip/OpenAI SDK와 같은 종류의 문제).
2. FLAGS_use_mkldnn=0 / enable_mkldnn=False: 이 PC의 CPU 추론 환경에서 oneDNN(MKL-DNN)
   백엔드를 쓰면 "NotImplementedError: ConvertPirAttribute2RuntimeAttribute not
   support..." 에러로 크래시한다(paddlepaddle 3.x의 PIR 실행기와 oneDNN 조합 버그로
   추정). oneDNN을 끄면 정상 동작 확인됨.

운영 서버(폐쇄망, Linux)에 실제로 배포할 때는 이 두 우회 설정이 그대로 필요한지,
아니면 이 PC(Windows)에서만 나는 문제인지 재확인이 필요하다.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("SSL_CERT_FILE", str(Path(__file__).resolve().parent.parent / "combined_ca_bundle.pem"))
os.environ.setdefault("REQUESTS_CA_BUNDLE", str(Path(__file__).resolve().parent.parent / "combined_ca_bundle.pem"))
os.environ.setdefault("FLAGS_use_mkldnn", "0")

from paddleocr import PaddleOCR  # noqa: E402

TEXT_PER_PAGE_THRESHOLD = 20  # 이 미만이면 사실상 텍스트 인식 실패로 판단(참고용)


def main():
    if len(sys.argv) != 2:
        print(f"사용법: python {Path(sys.argv[0]).name} <이미지 또는 PDF 경로>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"파일을 찾을 수 없습니다: {path}")
        sys.exit(1)

    print("OCR 모델 로딩 중... (최초 1회는 모델 다운로드로 시간이 걸릴 수 있음)")
    ocr = PaddleOCR(
        lang="korean",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )

    results = ocr.predict(str(path))
    total_chars = 0
    total_lines = 0
    for page_idx, res in enumerate(results, start=1):
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        print(f"\n--- {page_idx}페이지 (인식된 줄: {len(texts)}) ---")
        for text, score in zip(texts, scores):
            print(f"[{score:.3f}] {text}")
            total_chars += len(text)
            total_lines += 1

    print("\n" + "=" * 50)
    print(f"결과: 총 {total_lines}줄, {total_chars}자 인식")
    if total_chars < TEXT_PER_PAGE_THRESHOLD:
        print("→ 인식된 글자가 거의 없습니다. 스캔 품질이 낮거나 빈 페이지일 수 있습니다.")


if __name__ == "__main__":
    main()
