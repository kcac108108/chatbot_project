const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = await context.newPage();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('http://127.0.0.1:8000/', { waitUntil: 'networkidle' });

  await page.locator('#menu-toggle').click();
  await page.waitForTimeout(300);
  await page.locator('#keyword-input').fill('SSL 인증서');
  await page.locator('#keyword-search-btn').click();
  await page.waitForSelector('#typing-indicator', { state: 'detached', timeout: 30000 }).catch(() => {});
  await page.waitForTimeout(500);

  // close drawer by clicking the overlay/backdrop if present
  const overlay = page.locator('.sidebar-overlay, .drawer-overlay, #sidebar-backdrop').first();
  if (await overlay.count() > 0) {
    await overlay.click({ force: true }).catch(() => {});
  } else {
    // fallback: click menu-toggle again
    await page.locator('#menu-toggle').click().catch(() => {});
  }
  await page.waitForTimeout(300);

  await page.screenshot({ path: 'mobile_kw_result2.png', fullPage: true });

  const lastBubble = page.locator('.msg-row.bot').last();
  const text = await lastBubble.locator('.bubble').first().innerText().catch(() => '(no bubble found)');
  console.log('LAST BOT TEXT:', text);

  await browser.close();
})();
