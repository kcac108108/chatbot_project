// "빠른 검색" 사이드바 입력창 테스트 — 키워드를 입력하고 버튼/Enter로 검색, 결과 확인.
// 사용법: node cdp_keyword_search.js "키워드" 스크린샷파일명.png [via-enter|scope:메뉴명]
const { chromium } = require('playwright');

(async () => {
  const keyword = process.argv[2];
  const shotName = process.argv[3] || 'keyword_search.png';
  const action = process.argv[4];
  const viaEnter = action === 'via-enter';

  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const page = context.pages()[0];

  await page.reload();
  await page.waitForSelector('#keyword-input');

  if (action && action.startsWith('scope:')) {
    const label = action.slice('scope:'.length);
    await page.locator('.side-item', { hasText: label }).first().click();
    await page.waitForTimeout(300);
    console.log(`스코프 전환: ${label}`);
  }

  await page.fill('#keyword-input', keyword);
  if (viaEnter) {
    await page.press('#keyword-input', 'Enter');
  } else {
    await page.click('#keyword-search-btn');
  }
  await page.waitForSelector('#typing-indicator', { state: 'detached', timeout: 20000 }).catch(() => {});
  await page.waitForTimeout(500);

  const lastBubble = await page.locator('.msg-row.bot .bubble').last().textContent();
  console.log('마지막 봇 응답:', lastBubble);

  const sourceCount = await page.locator('.msg-row.bot .source-row').count();
  console.log('근거자료 카드 수(마지막 메시지 포함 전체):', sourceCount);

  await page.screenshot({ path: shotName, fullPage: true });
  console.log(`스크린샷 저장: ${shotName}`);

  await browser.close();
})();
