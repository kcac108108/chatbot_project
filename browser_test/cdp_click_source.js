const { chromium } = require('playwright');
(async () => {
  const shot = process.argv[2] || 'source_click.png';
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];
  const detailHead = page.locator('.detail-head').first();
  if (await detailHead.count()) {
    await detailHead.click();
    await page.waitForTimeout(400);
  }
  const chip = page.locator('.source-chip').first();
  await chip.click();
  await page.waitForTimeout(500);
  await page.screenshot({ path: shot, fullPage: true });
  console.log('소스 칩 클릭 완료');
  await browser.close();
})();
