const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];
  let dialogFired = false;
  page.on('dialog', async d => { dialogFired = true; console.log('[다이얼로그 발생!]', d.message()); await d.dismiss(); });

  const input = page.locator('textarea, input[type="text"]').first();
  const sendBtn = page.locator('#send');
  const payload = '<img src=x onerror=alert(1)>한국환경공단';
  await input.fill(payload);
  await input.press('Enter');
  await sendBtn.waitFor({ state: 'attached' });
  await page.waitForFunction(() => !document.getElementById('send').disabled, { timeout: 40000 }).catch(() => {});
  await page.waitForTimeout(1000);

  console.log('alert 다이얼로그 발생 여부:', dialogFired);

  // 질문 말풍선에 실제로 <img> 태그가 DOM 요소로 생성됐는지 확인(innerHTML로 꽂혔다면 요소가 생김)
  const imgCount = await page.locator('.msg-row.user img').count();
  console.log('사용자 말풍선 안 <img> 요소 개수:', imgCount);

  // 말풍선에 텍스트가 어떻게 보이는지(이스케이프돼서 글자 그대로 보이는지)
  const bubbleText = await page.locator('.msg-row.user').last().textContent();
  console.log('말풍선 텍스트:', bubbleText);

  await page.screenshot({ path: 'xss_check.png', fullPage: true });
  await browser.close();
})();
