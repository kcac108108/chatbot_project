const { chromium } = require('playwright');
(async () => {
  const [, , shotfile] = process.argv;
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];
  const size = page.viewportSize();
  console.log('current viewport:', JSON.stringify(size));
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(500);
  const size2 = page.viewportSize();
  console.log('after set:', JSON.stringify(size2));
  await page.screenshot({ path: shotfile || 'check.png', fullPage: false });
  await browser.close();
})();
