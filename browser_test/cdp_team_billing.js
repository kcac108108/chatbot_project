const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = await context.newPage();
  await page.goto('http://127.0.0.1:8000/', { waitUntil: 'networkidle' });

  const input = page.locator('#question');
  await input.fill('대한무역투자진흥공사 2026 글로벌공급망팀 청구내역 알려줘');
  await page.locator('#send').click();

  await page.waitForSelector('#typing-indicator', { state: 'detached', timeout: 30000 }).catch(() => {});
  await page.waitForTimeout(500);

  const lastBubble = page.locator('.msg-row.bot').last();
  const text = await lastBubble.locator('.bubble').first().innerText().catch(() => '(no bubble found)');
  console.log('LAST BOT TEXT:', text);

  const sourceCount = await page.locator('.source-row').count();
  console.log('SOURCE ROWS:', sourceCount);

  await page.screenshot({ path: 'team_billing_result.png', fullPage: true });

  await browser.close();
})();
