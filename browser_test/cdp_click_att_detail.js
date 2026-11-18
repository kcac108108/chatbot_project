const { chromium } = require('playwright');
(async () => {
  const [, , shotfile] = process.argv;
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];
  // click the "상세" button specifically inside the attachment mode-toggle row
  const btn = page.locator('.attachment-mode-btn, .attachment-mode-btns button').filter({ hasText: '상세' }).first();
  await btn.click();
  await page.waitForTimeout(3000); // GPT generation may take a few seconds
  if (shotfile) await page.screenshot({ path: shotfile, fullPage: true });
  console.log('clicked attachment detail toggle');
  await browser.close();
})();
