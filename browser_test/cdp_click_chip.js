// Usage: node cdp_click_chip.js "<칩에 포함된 텍스트>" shotfile.png
const { chromium } = require('playwright');
(async () => {
  const [, , label, shotfile] = process.argv;
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];
  const chip = page.locator('.source-chip, .attachment-chip, button, a').filter({ hasText: label }).first();
  await chip.click();
  await page.waitForTimeout(800);
  if (shotfile) {
    await page.screenshot({ path: shotfile, fullPage: true });
  }
  console.log(`클릭: ${label}`);
  await browser.close();
})();
