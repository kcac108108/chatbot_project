const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(500);
  await page.screenshot({ path: 'mobile_1.png', fullPage: false });
  console.log('모바일 뷰포트 스크린샷 저장');
  // 사이드바 햄버거 메뉴가 있으면 클릭
  const hamburger = page.locator('[class*="hamburger"], button[aria-label*="메뉴"], .menu-toggle').first();
  if (await hamburger.count()) {
    await hamburger.click();
    await page.waitForTimeout(400);
    await page.screenshot({ path: 'mobile_2_sidebar.png', fullPage: false });
    console.log('사이드바 열림 스크린샷 저장');
  } else {
    console.log('햄버거 버튼 못 찾음');
  }
  await browser.close();
})();
