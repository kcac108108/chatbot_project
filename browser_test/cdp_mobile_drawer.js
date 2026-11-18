const { chromium } = require('playwright');
(async () => {
  const [, , shotfile] = process.argv;
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('#menu-toggle').click();
  await page.waitForTimeout(500);
  await page.screenshot({ path: shotfile || 'drawer.png', fullPage: false });
  console.log('drawer opened');
  await browser.close();
})();
