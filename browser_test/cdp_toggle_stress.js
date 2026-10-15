const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];
  await page.goto('http://127.0.0.1:8000/');
  await page.waitForSelector('text=이력관리시스템');

  async function ask(q) {
    const input = page.locator('textarea, input[type="text"]').first();
    const sendBtn = page.locator('#send');
    await input.fill(q);
    await input.press('Enter');
    await sendBtn.waitFor({ state: 'attached' });
    await page.waitForFunction(() => !document.getElementById('send').disabled, { timeout: 40000 }).catch(() => {});
    await page.waitForTimeout(800);
  }

  await ask('한국환경공단 기관정보 알려줘');
  const histBtn = page.locator('button', { hasText: '이력 보기' }).first();
  await histBtn.click();
  await page.waitForTimeout(1200);

  const rows = page.locator('.history-row');
  const n = await rows.count();
  console.log('행 개수:', n);

  // 같은 행을 열었다 닫았다 10번 빠르게 반복(토글 버그는 "같은 요소를 빠르게 다시 토글"할 때 주로 재현되는 패턴)
  let failures = 0;
  for (let i = 0; i < 10; i++) {
    await rows.nth(0).click();
    await page.waitForTimeout(80);
    const isOpen = await rows.nth(0).evaluate(el => el.classList.contains('open'));
    const expected = i % 2 === 0; // 짝수번째 클릭 후 열림 기대
    if (isOpen !== expected) {
      failures++;
      console.log(`[불일치] ${i}번째 클릭 후 open=${isOpen}, 기대=${expected}`);
    }
  }
  console.log('불일치 횟수:', failures, '/ 10');

  // 서로 다른 행 5개를 순서 섞어서(0,3,1,4,2) 클릭
  const order = [0, 3, 1, 4, 2];
  for (const idx of order) {
    if (idx < n) {
      await rows.nth(idx).click();
      await page.waitForTimeout(100);
    }
  }
  const openStates = [];
  for (let i = 0; i < Math.min(5, n); i++) {
    openStates.push(await rows.nth(i).evaluate(el => el.classList.contains('open')));
  }
  console.log('순서 섞어 클릭 후 open 상태:', openStates);

  await page.screenshot({ path: 'toggle_stress.png', fullPage: true });
  await browser.close();
  console.log('완료');
})();
