const { chromium } = require('playwright');
(async () => {
  const shot = process.argv[2] || 'scroll_shot.png';
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];
  await page.setViewportSize({ width: 1280, height: 1400 });
  await page.waitForTimeout(300);
  // 스크롤 가능한 채팅 영역을 맨 아래로
  await page.evaluate(() => {
    const el = document.getElementById('chat') || document.querySelector('[id*="chat"]');
    if (el) el.scrollTop = el.scrollHeight;
  });
  await page.waitForTimeout(300);
  await page.screenshot({ path: shot, fullPage: false });
  console.log('스크롤 후 스크린샷 저장');
  await browser.close();
})();
