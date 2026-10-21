const { chromium } = require('playwright');
(async () => {
  const mode = process.argv[2];
  const shotName = process.argv[3];
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];
  const btn = page.locator('.attachment-mode-btn').filter({ hasText: mode }).last();
  await btn.click();
  await page.waitForTimeout(6000);
  const chat = page.locator('#chat');
  await chat.evaluate(el => el.scrollTo(0, el.scrollHeight));
  await page.waitForTimeout(300);
  await page.screenshot({ path: shotName, fullPage: false });
  console.log('완료:', shotName);
  await browser.close();
})();
