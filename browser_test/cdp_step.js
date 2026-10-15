// CDP로 떠있는 브라우저에 접속해서 질문 하나를 입력하고 결과를 확인.
// 사용법: node cdp_step.js "질문 내용" 스크린샷파일명.png [scope|click:텍스트]
const { chromium } = require('playwright');

(async () => {
  const question = process.argv[2];
  const shotName = process.argv[3] || 'step.png';
  const action = process.argv[4]; // 예: "scope:심의위원회" 또는 "click:이력 보기"

  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];

  if (action && action.startsWith('scope:')) {
    const label = action.slice('scope:'.length);
    await page.locator('.side-item', { hasText: label }).first().click();
    await page.waitForTimeout(300);
    console.log(`스코프 전환: ${label}`);
  } else if (action && action.startsWith('click:')) {
    const label = action.slice('click:'.length);
    await page.locator('button', { hasText: label }).first().click();
    await page.waitForTimeout(1200);
    console.log(`버튼 클릭: ${label}`);
  }

  if (question) {
    const input = page.locator('textarea, input[type="text"]').first();
    const sendBtn = page.locator('#send');
    await input.fill(question);
    await input.press('Enter');
    await sendBtn.waitFor({ state: 'attached' });
    await page.waitForFunction(() => !document.getElementById('send').disabled, { timeout: 40000 }).catch(() => {});
    await page.waitForTimeout(800);
    console.log(`질문: ${question}`);
  }

  await page.screenshot({ path: shotName, fullPage: true });
  console.log(`스크린샷 저장: ${shotName}`);

  await browser.close(); // CDP 연결만 닫음, 실제 브라우저는 안 죽음
})();
