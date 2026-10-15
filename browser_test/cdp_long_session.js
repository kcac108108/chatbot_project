const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];
  await page.goto('http://127.0.0.1:8000/');
  await page.waitForSelector('text=이력관리시스템');

  page.on('console', msg => { if (msg.type() === 'error') console.log('[콘솔 에러]', msg.text()); });
  page.on('pageerror', err => console.log('[페이지 에러]', err.message));

  const questions = [
    '한국환경공단 관련 이력 알려줘', '수수료는 얼마야?', '담당자는 누구야?',
    '한국철강협회 전체 이력 보여줘', '거기서 제일 최근 거 자세히 알려줘',
    '박상용 계장이 요청한 거 있어?', '그 중에 2023년 건만 보여줘',
    '심의위원회에서 제공불가된 안건을 알려줘', '그 중에 가장 최근 거는?',
    '내부자료 담당자 목록 알려줘', '국가 항목을 요청한 업체를 알려줘',
    '외부요청으로 등록된 업체명을 알려줘', '한국은행이 제공한 자료 있어?',
    '대한무역투자진흥공사 전체 이력 알려줘', '한국무역보험공사 이력 알려줘',
    '고마워', '안녕', '첨부파일 중에 보안 점검 관련 내용이 있을까?',
    '그거 자세히 알려줘', '수고했어',
  ];

  const start = Date.now();
  for (let i = 0; i < questions.length; i++) {
    const input = page.locator('textarea, input[type="text"]').first();
    const sendBtn = page.locator('#send');
    await input.fill(questions[i]);
    await input.press('Enter');
    await sendBtn.waitFor({ state: 'attached' });
    await page.waitForFunction(() => !document.getElementById('send').disabled, { timeout: 40000 }).catch(() => {});
    console.log(`[${i + 1}/${questions.length}] ${questions[i]} 완료 (${((Date.now() - start) / 1000).toFixed(1)}s 경과)`);
  }

  const msgCount = await page.locator('.msg-row').count();
  console.log('최종 msg-row 개수:', msgCount, '(기대: 1(인사말) + 20*2 =', 1 + questions.length * 2, ')');

  await page.screenshot({ path: 'long_session_final.png', fullPage: true });
  await browser.close();
  console.log('완료');
})();
