const { chromium } = require('playwright');
(async () => {
  const [, , shotfile] = process.argv;
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];
  await page.evaluate(() => {
    const chat = document.querySelector('#chat');
    if (chat) chat.scrollTop = chat.scrollHeight;
  });
  await page.waitForTimeout(500);
  await page.screenshot({ path: shotfile || 'scroll.png', fullPage: true });
  console.log('scrolled');
  await browser.close();
})();
