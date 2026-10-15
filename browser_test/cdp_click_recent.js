const { chromium } = require('playwright');
(async () => {
  const label = process.argv[2];
  const shot = process.argv[3] || 'recent_click.png';
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];
  const items = await page.locator('.side-section', { hasText: '최근 검색' }).locator('button, .side-item').allTextContents();
  console.log('최근검색 목록:', items);
  await page.locator('.side-section', { hasText: '최근 검색' }).locator('button, .side-item', { hasText: label }).first().click();
  await page.waitForTimeout(600);
  await page.screenshot({ path: shot, fullPage: true });
  console.log(`최근검색 클릭: ${label}`);
  await browser.close();
})();
