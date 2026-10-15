// 영구 브라우저 세션 시작 — CDP 포트를 열어두고 계속 떠 있는다.
// 이후 cdp_step.js로 한 번에 한 질문씩 즉흥적으로 보내고 결과를 확인한다.
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ args: ['--remote-debugging-port=9222'] });
  const page = await browser.newPage();
  page.on('console', msg => { if (msg.type() === 'error') console.log('[콘솔 에러]', msg.text()); });
  page.on('pageerror', err => console.log('[페이지 에러]', err.message));
  page.on('response', res => { if (res.status() >= 400) console.log('[HTTP 에러]', res.status(), res.url()); });
  await page.goto('http://127.0.0.1:8000/');
  await page.waitForSelector('text=이력관리시스템');
  console.log('READY');
  // 프로세스를 계속 살려둠
  await new Promise(() => {});
})();
