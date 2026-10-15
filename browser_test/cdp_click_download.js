const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];
  const attBtn = page.locator('button', { hasText: '.sql' }).first();
  if (await attBtn.count()) {
    await attBtn.click();
    await page.waitForTimeout(500);
  }
  const link = page.locator('a.attachment-download-link, a', { hasText: '파일 다운로드' }).first();
  console.log('링크 존재:', await link.count());
  if (await link.count()) {
    console.log('href:', await link.getAttribute('href'));
    const [download] = await Promise.all([
      page.waitForEvent('download', { timeout: 10000 }).catch(e => null),
      link.click(),
    ]);
    if (download) {
      console.log('다운로드 성공:', download.suggestedFilename());
      const path = await download.path();
      const fs = require('fs');
      console.log('파일 크기:', fs.statSync(path).size, 'bytes');
    } else {
      console.log('다운로드 이벤트 없음 (새 탭/직접 네비게이션일 수 있음)');
    }
  }
  await browser.close();
})();
