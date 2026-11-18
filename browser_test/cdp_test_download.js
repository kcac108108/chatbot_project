const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];
  const downloadPromise = page.waitForEvent('download', { timeout: 15000 }).catch(() => null);
  const btn = page.locator('text=파일 다운로드').first();
  await btn.click();
  const download = await downloadPromise;
  if (download) {
    console.log('다운로드 성공:', await download.suggestedFilename());
  } else {
    console.log('다운로드 이벤트 발생 안 함');
  }
  await browser.close();
})();
