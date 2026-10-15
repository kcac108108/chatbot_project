const { chromium } = require('playwright');
(async () => {
  const label = process.argv[2];
  const shot = process.argv[3] || 'faq_click.png';
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];
  const sendBtn = page.locator('#send');
  await page.locator('.faq-item', { hasText: label }).first().click();
  await sendBtn.waitFor({ state: 'attached' });
  await page.waitForFunction(() => !document.getElementById('send').disabled, { timeout: 40000 }).catch(() => {});
  await page.waitForTimeout(800);
  await page.screenshot({ path: shot, fullPage: true });
  console.log(`FAQ 클릭: ${label}`);
  await browser.close();
})();
