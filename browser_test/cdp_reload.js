const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];
  await page.reload();
  await page.waitForSelector('text=이력관리시스템');
  console.log('새로고침 완료 — 새 대화 시작');
  await browser.close();
})();
