const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];
  const input = page.locator('textarea, input[type="text"]').first();
  const sendBtn = page.locator('#send');
  // Shift+Enter로 줄바꿈 삽입 후 전송
  await input.fill('한국환경공단');
  await input.press('Shift+Enter');
  await input.type('관련 이력');
  await input.press('Shift+Enter');
  await input.type('알려줘');
  await input.press('Enter');
  await sendBtn.waitFor({ state: 'attached' });
  await page.waitForFunction(() => !document.getElementById('send').disabled, { timeout: 40000 }).catch(() => {});
  await page.waitForTimeout(800);
  await page.screenshot({ path: 'multiline_input.png', fullPage: true });
  console.log('완료');
  await browser.close();
})();
