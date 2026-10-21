// 첨부 칩을 펼친 뒤, 지정된 모드 버튼("요약" 또는 "상세")을 클릭하고 스크린샷
const { chromium } = require('playwright');
(async () => {
  const mode = process.argv[2] || null; // "요약" | "상세" | null(펼치기만)
  const shotName = process.argv[3] || 'att_detail.png';

  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];

  const attChip = page.locator('.attachment-chip').last();
  await attChip.click();
  await page.waitForTimeout(300);

  if (mode) {
    const modeBtn = page.locator('.attachment-mode-btn').filter({ hasText: mode }).last();
    await modeBtn.click();
    await page.waitForTimeout(4000); // 상세 생성은 GPT 호출이라 시간이 걸릴 수 있음
  }

  const chat = page.locator('#chat');
  await chat.evaluate(el => el.scrollTo(0, el.scrollHeight));
  await page.waitForTimeout(300);
  await page.screenshot({ path: shotName, fullPage: false });
  console.log('완료:', shotName);
  await browser.close();
})();
