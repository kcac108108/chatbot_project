// "빠른 검색" 여러 키워드 연속 테스트 — 콘솔 에러/HTTP 에러 감시하며 UI로 직접 검색.
const { chromium } = require('playwright');

const KEYWORDS = [
  '환경',
  '수출입 통계',
  'MOU',
  '보안 점검',
  '계약 해지',
  '원산지',
  '전자상거래',
  '통관',
  '개인정보',
  '지식재산권',
];

(async () => {
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];

  const errors = [];
  page.on('console', msg => { if (msg.type() === 'error') errors.push('[콘솔] ' + msg.text()); });
  page.on('pageerror', err => errors.push('[페이지] ' + err.message));
  page.on('response', res => { if (res.status() >= 400) errors.push('[HTTP ' + res.status() + '] ' + res.url()); });

  await page.reload();
  await page.waitForSelector('#keyword-input');

  for (const kw of KEYWORDS) {
    await page.fill('#keyword-input', kw);
    await page.click('#keyword-search-btn');
    await page.waitForSelector('#typing-indicator', { state: 'detached', timeout: 20000 }).catch(() => {});
    await page.waitForTimeout(300);
    const bubble = await page.locator('.msg-row.bot .bubble').last().textContent();
    console.log(`"${kw}" -> ${bubble}`);
  }

  console.log('--- 에러 목록 ---');
  if (errors.length === 0) console.log('없음');
  else errors.forEach(e => console.log(e));

  await page.screenshot({ path: 'shot_kw_batch.png', fullPage: true });
  await browser.close();
})();
