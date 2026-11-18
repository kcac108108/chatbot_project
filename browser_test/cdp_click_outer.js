const { chromium } = require('playwright');
(async () => {
  const [, , shotfile] = process.argv;
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];
  // click the first detail-head labeled "상세" (the outer record-detail chip)
  const chip = page.locator('.detail-head, .source-row').filter({ hasText: '상세' }).first();
  await chip.click();
  await page.waitForTimeout(600);
  if (shotfile) await page.screenshot({ path: shotfile, fullPage: true });
  console.log('clicked outer detail');
  await browser.close();
})();
