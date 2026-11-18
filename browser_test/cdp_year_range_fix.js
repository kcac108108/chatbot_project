const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = await context.newPage();
  await page.goto('http://127.0.0.1:8000/', { waitUntil: 'networkidle' });

  const input = page.locator('#question');
  await input.fill('최근 5개년 철강협회 심의결과 알려줘');
  await page.locator('#send').click();

  await page.waitForSelector('#typing-indicator', { state: 'detached', timeout: 30000 }).catch(() => {});
  await page.waitForTimeout(500);

  const lastBubble = page.locator('.msg-row.bot').last();
  const text = await lastBubble.locator('.bubble').first().innerText().catch(() => '(no bubble found)');
  console.log('LAST BOT TEXT:', text);

  const sourceCount = await page.locator('.source-row').count();
  console.log('SOURCE ROWS:', sourceCount);
  for (let i = 0; i < sourceCount; i++) {
    const chipText = await page.locator('.source-row').nth(i).locator('.source-chip, .source-row > *').first().innerText().catch(() => '');
    console.log('  SOURCE', i, chipText);
  }

  await page.screenshot({ path: 'year_range_fix_result.png', fullPage: true });

  await browser.close();
})();
