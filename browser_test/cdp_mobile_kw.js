const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = await context.newPage();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('http://127.0.0.1:8000/', { waitUntil: 'networkidle' });

  // open hamburger drawer
  const hamburger = page.locator('#menu-toggle').first();
  if (await hamburger.count() > 0) {
    await hamburger.click();
    await page.waitForTimeout(300);
  }

  await page.screenshot({ path: 'mobile_kw_drawer.png' });

  const input = page.locator('#keyword-input');
  await input.fill('FTA');
  const btn = page.locator('#keyword-search-btn');
  await btn.click();

  await page.waitForSelector('#typing-indicator', { state: 'detached', timeout: 30000 }).catch(() => {});
  await page.waitForTimeout(500);

  await page.screenshot({ path: 'mobile_kw_result.png', fullPage: true });

  const lastBubble = page.locator('.msg-row.bot').last();
  const text = await lastBubble.locator('.bubble').first().innerText().catch(() => '(no bubble found)');
  console.log('LAST BOT TEXT:', text);

  const sourceCount = await page.locator('.source-row').count();
  console.log('SOURCE ROWS:', sourceCount);

  await browser.close();
})();
