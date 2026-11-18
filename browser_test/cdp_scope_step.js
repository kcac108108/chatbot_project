// Usage: node cdp_scope_step.js "<사이드바 빠른메뉴 라벨>" "<질문>" shotfile.png
const { chromium } = require('playwright');

(async () => {
  const [, , scopeLabel, question, shotfile] = process.argv;
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];

  if (scopeLabel) {
    const btn = page.locator('.sidebar button, .sidebar li').filter({ hasText: scopeLabel }).first();
    await btn.click();
    await page.waitForTimeout(300);
  }

  const input = page.locator('textarea, input[type="text"]').first();
  const sendBtn = page.locator('#send');
  await input.fill(question);
  await input.press('Enter');
  await sendBtn.waitFor({ state: 'attached' });
  await page.waitForFunction(() => !document.getElementById('send').disabled, { timeout: 40000 }).catch(() => {});
  await page.waitForTimeout(800);

  if (shotfile) {
    await page.screenshot({ path: shotfile, fullPage: true });
  }
  console.log(`스코프: ${scopeLabel}, 질문: ${question}`);
  await browser.close();
})();
