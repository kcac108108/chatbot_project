const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];

  const pageA = await context.newPage();
  const pageB = await context.newPage();
  await pageA.goto('http://127.0.0.1:8000/');
  await pageB.goto('http://127.0.0.1:8000/');
  await pageA.waitForSelector('text=이력관리시스템');
  await pageB.waitForSelector('text=이력관리시스템');

  async function ask(page, q) {
    const input = page.locator('textarea, input[type="text"]').first();
    const sendBtn = page.locator('#send');
    await input.fill(q);
    await input.press('Enter');
    await sendBtn.waitFor({ state: 'attached' });
    await page.waitForFunction(() => !document.getElementById('send').disabled, { timeout: 40000 }).catch(() => {});
    await page.waitForTimeout(800);
  }

  // 탭 A: 한국환경공단 이야기
  await ask(pageA, '한국환경공단 관련 이력 알려줘');
  // 탭 B: 전혀 다른 주제
  await ask(pageB, '한국철강협회 관련 이력 알려줘');
  // 탭 A에서 "그거" 대명사로 후속질문 — 탭 B의 대화에 영향 안 받아야 함
  await ask(pageA, '거기 수수료는 얼마야?');

  const answerA = await pageA.locator('.msg-row.bot').last().textContent();
  console.log('탭A 최종 답변:', answerA);

  await pageA.screenshot({ path: 'tabA_final.png', fullPage: true });
  await pageB.screenshot({ path: 'tabB_final.png', fullPage: true });

  await pageA.close();
  await pageB.close();
  await browser.close();
  console.log('완료');
})();
