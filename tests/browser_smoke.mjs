/** Real-browser regressions against an isolated, simulated-model preview. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp } from 'node:fs/promises';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const scratch = await mkdtemp(path.join(tmpdir(), 'transparent-browser-'));
await mkdir(path.join(root, 'test-results'), { recursive: true });
const artifacts = await mkdtemp(path.join(root, 'test-results', 'browser-'));
const listener = createServer();
await new Promise(resolve => listener.listen(0, '127.0.0.1', resolve));
const port = listener.address().port;
await new Promise(resolve => listener.close(resolve));
const base = `http://127.0.0.1:${port}`;
const python = process.env.PYTHON_BINARY || path.join(root, '.venv', 'bin', 'python');
const server = spawn(python, ['-m', 'uvicorn', 'tests.ui_preview:app', '--host', '127.0.0.1', '--port', String(port), '--no-access-log'], {
  cwd: root,
  env: { ...process.env, CHAT_DB_PATH: path.join(scratch, 'chat.db') },
  stdio: ['ignore', 'pipe', 'pipe'],
});
let serverOutput = '';
server.stdout.on('data', data => { serverOutput += data; });
server.stderr.on('data', data => { serverOutput += data; });
let startupError;
server.on('error', error => { startupError = error; });

const checks = [];
const check = (name, condition) => {
  assert.ok(condition, name);
  checks.push(name);
  console.log(`PASS ${name}`);
};
const settle = async page => {
  await page.waitForFunction(() => !document.querySelector('.htmx-request'));
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
};
const finished = async page => {
  await settle(page);
  await page.waitForFunction(() => !document.querySelector('[data-generation-status="queued"], [data-generation-status="running"]'));
  await settle(page);
};
const fit = async page => page.evaluate(() => {
  const selectors = ['.chat-header', '.context-bar', '#message-list', '.composer-wrap'];
  return document.documentElement.scrollWidth <= innerWidth && selectors.every(selector => {
    const element = document.querySelector(selector);
    if (!element) return true;
    const rect = element.getBoundingClientRect();
    return rect.left >= 0 && rect.right <= innerWidth + 1 && rect.top >= 0 && rect.bottom <= innerHeight + 1;
  });
});

let browser;
try {
  for (let attempt = 0; attempt < 100; attempt++) {
    if (startupError) throw startupError;
    if (server.exitCode !== null) throw new Error(`Preview exited: ${serverOutput}`);
    try { if ((await fetch(base)).ok) break; } catch { /* Wait for the listening socket. */ }
    if (attempt === 99) throw new Error(`Preview startup timed out: ${serverOutput}`);
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  const systemChromium = '/Applications/Chromium.app/Contents/MacOS/Chromium';
  const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || (existsSync(systemChromium) ? systemChromium : undefined);
  browser = await chromium.launch({ headless: process.env.HEADED !== '1', executablePath });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, locale: 'en-US', timezoneId: 'Asia/Seoul' });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('dialog', dialog => dialog.dismiss());
  await page.goto(base);
  await page.screenshot({ path: path.join(artifacts, 'welcome.png'), scale: 'css' });
  await page.getByRole('button', { name: 'Start a conversation' }).click();
  await settle(page);
  const sessionId = await page.locator('#active-session').inputValue();
  const chatURL = page.url();
  check('first-use flow creates a chat', !!sessionId);
  check('empty chat has a distinguishable numbered title', await page.getByLabel('Chat title', { exact: true }).inputValue() === `New chat #${sessionId}`);
  check('CSS and JavaScript URLs are versioned', await page.locator('link[rel=stylesheet]').getAttribute('href').then(href => href.includes('?v=')));

  const firstInput = '[slow] Browser stop regression\n  한글🙂 A long first message that should be truncated automatically without asking a model';
  const automaticTitle = Array.from(firstInput.replace(/\s+/g, ' ').trim()).slice(0, 59).join('').trimEnd() + '…';
  await page.locator('.composer textarea').fill(firstInput);
  await page.locator('.composer textarea').press('Enter');
  await page.locator('.streaming .content span').waitFor();
  await settle(page);
  const sentAt = await page.locator('.session-item.active time').getAttribute('datetime');
  check('first message becomes a normalized truncated title', await page.getByLabel('Chat title', { exact: true }).inputValue() === automaticTitle);
  check('sidebar title excludes date metadata', await page.locator('.session-item.active .session-title').textContent() === automaticTitle);
  const localDate = new Intl.DateTimeFormat('en-US', {timeZone: 'Asia/Seoul', year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'}).format(new Date(sentAt));
  check('last-modified date uses local time from a UTC timestamp', sentAt.endsWith('+00:00') && await page.locator('.session-item.active time').textContent() === `Updated ${localDate}`);
  await page.locator('.btn-stop').click();
  await settle(page);
  check('Stop restores Send', await page.locator('.composer button[type=submit]').isEnabled());
  check('Stop button is hidden after cancellation', !await page.locator('.btn-stop').isVisible());
  check('Stopped label survives htmx settling', await page.locator('.generation-label').textContent() === 'Stopped');
  check('partial output is retained', !!await page.locator('.generation-error .content').textContent());
  await page.screenshot({ path: path.join(artifacts, 'stopped.png'), scale: 'css' });

  await page.locator('.composer textarea').fill('[fail] Retry regression');
  await page.locator('.composer textarea').press('Enter');
  await finished(page);
  await page.locator('.retry-generation').last().click();
  await finished(page);
  check('retry finishes without duplicating the response', await page.locator('.message[data-message-role="assistant"]').count() === 1);
  check('literal model HTML remains inert', await page.evaluate(() => !window.unsafeExecuted));
  check('output indentation is preserved', (await page.locator('.message[data-message-role="assistant"] .content').textContent()).includes('\n    indented  code'));
  check('later messages leave the first-message title intact', await page.getByLabel('Chat title', { exact: true }).inputValue() === automaticTitle);
  check('completed response refreshes sidebar modification time', new Date(await page.locator('.session-item.active time').getAttribute('datetime')) > new Date(sentAt));

  await page.getByLabel('Chat title', { exact: true }).fill('My manually named project');
  await page.getByLabel('Chat title', { exact: true }).press('Tab');
  await settle(page);
  await page.locator('.composer textarea').fill('[instant] Keep my manual title');
  await page.locator('.composer textarea').press('Enter');
  await finished(page);
  check('manual title survives a new message', await page.locator('.session-item.active .session-title').textContent() === 'My manually named project');
  await page.getByLabel('Chat title', { exact: true }).fill('');
  await page.getByLabel('Chat title', { exact: true }).press('Enter');
  await settle(page);
  check('clearing title with Enter immediately restores automatic label', await page.getByLabel('Chat title', { exact: true }).inputValue() === automaticTitle);
  await page.locator('#chat-search').fill('한글🙂');
  check('title search works with date metadata', await page.locator('.session-item:visible').count() === 1);
  await page.locator('#chat-search').fill('');
  await page.screenshot({ path: path.join(artifacts, 'titles-desktop.png'), scale: 'css' });

  const draft = 'Draft survives\npage changes';
  await page.locator('.composer textarea').fill(draft);
  await page.locator('.new-conversation').click();
  await settle(page);
  await page.locator(`.session-link[hx-get="/sessions/${sessionId}"]`).click();
  await settle(page);
  check('draft survives navigation', await page.locator('.composer textarea').inputValue() === draft);
  await page.reload();
  await settle(page);
  check('automatic title persists after reload', await page.getByLabel('Chat title', { exact: true }).inputValue() === automaticTitle);
  check('draft survives reload', await page.locator('.composer textarea').inputValue() === draft);

  await page.locator('.params-toggle > summary').click();
  await page.getByLabel('Max tokens', { exact: true }).fill('2048');
  await page.getByRole('button', { name: 'Apply', exact: true }).click();
  await settle(page);
  check('external Apply button saves the form', await page.getByLabel('Max tokens', { exact: true }).inputValue() === '2048');
  check('desktop regions fit', await fit(page));
  await page.screenshot({ path: path.join(artifacts, 'desktop.png'), scale: 'css' });

  for (let i = 0; i < 10; i++) {
    await page.locator('.new-conversation').click();
    await settle(page);
  }
  check('long session list starts limited to ten', await page.locator('.session-item:visible').count() === 10);
  await page.locator('.show-all').click();
  check('Show all reveals remaining sessions', await page.locator('.session-item:visible').count() === 12);
  await page.locator('.show-all').click();
  for (let i = 0; i < 3; i++) {
    await page.getByRole('button', { name: 'Create folder' }).click();
    await settle(page);
  }
  const folder = page.locator('.folder-item').last();
  await folder.locator('.folder-menu-btn').click();
  const menu = await folder.locator('.folder-menu').boundingBox();
  check('bottom folder menu stays inside viewport', menu.y >= 0 && menu.y + menu.height <= 900);
  // A real click also checks that the scroll container does not clip the item.
  await folder.locator('.folder-menu .danger').click();
  check('bottom folder Delete is clickable', true);
  await folder.locator('.folder-menu-btn').click();
  await page.screenshot({ path: path.join(artifacts, 'folder-menu.png'), scale: 'css' });
  await page.keyboard.press('Escape');
  check('Escape resets expanded state', await folder.locator('.folder-menu-btn').getAttribute('aria-expanded') === 'false');

  const mobile = await browser.newContext({ viewport: { width: 320, height: 640 }, isMobile: true, hasTouch: true });
  const mobilePage = await mobile.newPage();
  mobilePage.on('pageerror', error => errors.push(error.message));
  await mobilePage.goto(chatURL);
  await mobilePage.locator('.params-toggle > summary').tap();
  await mobilePage.locator('.advanced summary').tap();
  let apply = await mobilePage.getByRole('button', { name: 'Apply', exact: true }).boundingBox();
  check('Apply fits at 320px', apply.y >= 0 && apply.y + apply.height <= 640);
  check('settings header sticks while scrolling', await mobilePage.locator('.settings-heading').evaluate(el => getComputedStyle(el).position === 'sticky'));
  await mobilePage.locator('.settings-content').hover();
  await mobilePage.mouse.wheel(0, 500);
  await mobilePage.screenshot({ path: path.join(artifacts, 'mobile-settings.png'), scale: 'css' });
  await mobilePage.getByRole('button', { name: 'Apply', exact: true }).tap();
  await settle(mobilePage);
  check('mobile settings save', !await mobilePage.locator('.settings-content').isVisible());
  check('small mobile regions fit', await fit(mobilePage));
  await mobilePage.locator('.menu-btn').tap();
  await mobilePage.waitForFunction(() => document.querySelector('#sidebar').getBoundingClientRect().left >= -.1);
  await mobilePage.screenshot({ path: path.join(artifacts, 'mobile-drawer.png'), scale: 'css' });
  await mobilePage.keyboard.press('Escape');
  await mobilePage.waitForFunction(() => document.querySelector('#sidebar').getBoundingClientRect().right <= .1);
  check('Escape closes mobile drawer', await mobilePage.locator('.menu-btn').getAttribute('aria-expanded') === 'false');
  await mobilePage.setViewportSize({ width: 390, height: 844 });
  check('mobile regions fit at 390px', await fit(mobilePage));
  await mobilePage.screenshot({ path: path.join(artifacts, 'mobile-chat.png'), scale: 'css' });
  await page.locator('.new-conversation').click();
  await settle(page);
  await page.locator('.composer textarea').fill('[variants] Branch regression');
  await page.locator('.composer textarea').press('Enter');
  await finished(page);
  const firstAnswerId = await page.locator('.message[data-message-role="assistant"]').getAttribute('id');
  await page.getByRole('button', { name: 'Regenerate', exact: true }).click();
  await page.locator('.generation-progress').waitFor();
  check('regeneration keeps the existing response while running',
    (await page.locator(`#${firstAnswerId} .content`).textContent()).includes('version 1'));
  await finished(page);
  check('regeneration appends a selected response version', await page.locator('.variant-position').textContent() === '2/2');
  check('replacing a mutation button unlocks the composer', await page.locator('.composer button[type=submit]').isEnabled());
  await page.getByRole('button', { name: 'Previous response', exact: true }).click();
  await settle(page);
  check('previous response selects the original answer', await page.locator('.variant-position').textContent() === '1/2'
    && (await page.locator('.message[data-message-role="assistant"] .content').textContent()).includes('version 1'));
  await page.getByRole('button', { name: 'Next response', exact: true }).click();
  await settle(page);
  check('next response remains usable after the first switch', await page.locator('.variant-position').textContent() === '2/2');
  await page.getByRole('button', { name: 'Previous response', exact: true }).click();
  await settle(page);
  await page.locator('.composer textarea').fill('[instant] Continue selected version');
  await page.locator('.composer textarea').press('Enter');
  await finished(page);
  const prompt = await context.newPage();
  await prompt.goto(base + await page.locator('.message[data-message-role="assistant"] a').last().getAttribute('href'));
  const input = JSON.parse(await prompt.locator('#prompt-messages').textContent());
  check('follow-up context contains only the selected version',
    input.filter(message => message.role === 'assistant').map(message => message.content).join() === 'Response version 1 for this input.');
  await prompt.close();
  check('earlier-turn version controls lock after follow-up', await page.locator(`#${firstAnswerId} .variant-next`).isDisabled());
  await page.locator(`#${firstAnswerId} .fork-message`).click();
  await settle(page);
  check('intermediate fork keeps only the selected prefix', await page.locator('.message[data-message-role]').count() === 2
    && (await page.locator('.message[data-message-role="assistant"] .content').textContent()).includes('version 1')
    && await page.locator('.variant-position').textContent() === '1/1');
  page.removeAllListeners('dialog');
  page.once('dialog', dialog => dialog.accept());
  await page.locator('.delete-message:visible').click();
  await settle(page);
  page.on('dialog', dialog => dialog.dismiss());
  check('deleting the answer exposes an explicit response request', await page.getByRole('button', { name: 'Request response', exact: true }).isVisible());
  await page.getByRole('button', { name: 'Request response', exact: true }).click();
  await finished(page);
  check('requesting again does not duplicate the user message', await page.locator('.message[data-message-role="user"]').count() === 1
    && (await page.locator('.message[data-message-role="assistant"] .content').textContent()).includes('version 3'));
  await mobilePage.setViewportSize({ width: 320, height: 640 });
  await mobilePage.goto(page.url());
  await mobilePage.getByRole('button', { name: 'Regenerate', exact: true }).tap();
  await finished(mobilePage);
  await mobilePage.getByRole('button', { name: 'Previous response', exact: true }).tap();
  await settle(mobilePage);
  check('response controls work and fit at 320px', await mobilePage.locator('.variant-position').textContent() === '1/2'
    && await mobilePage.locator('.composer button[type=submit]').isEnabled() && await fit(mobilePage));
  await mobilePage.screenshot({ path: path.join(artifacts, 'mobile-response-versions.png'), scale: 'css' });
  await page.locator('.new-conversation').click();
  await settle(page);
  await page.locator('.input-tools-toggle').click();
  await page.locator('[name="web_search"]').check();
  const searchInput = '[multi-search] Compare the paper results';
  await page.locator('.composer textarea').fill(searchInput);
  await page.locator('.composer textarea').press('Enter');
  await page.locator('.tool-result[data-event-status="running"]').first().waitFor();
  check('tool calls and running search appear before the answer',
    await page.locator('.tool-call').count() === 1
    && await page.locator('.message[data-message-role="assistant"]').count() === 0
    && await page.locator('[sse-connect]').count() === 1);
  await finished(page);
  const searchSourceURL = page.url();
  const timeline = await page.locator('.message[data-message-role], .tool-event').evaluateAll(elements =>
    elements.map(element => element.dataset.messageRole || `event:${element.dataset.eventKind}`));
  check('conversation orders user, model call, separate results, and final answer',
    timeline.join() === 'user,event:assistant,event:tool,event:tool,assistant');
  await page.locator('.source-snippet summary').first().click();
  check('saved source snippets can be read in the result itself',
    await page.locator('.source-snippet p').first().isVisible()
    && (await page.locator('.source-snippet p').first().textContent()).includes('42 samples'));
  const toolPrompt = await context.newPage();
  await toolPrompt.goto(base + await page.locator('.tool-call a').first().getAttribute('href'));
  const roundOne = JSON.parse(await toolPrompt.locator('#prompt-messages').textContent());
  const toolSchema = JSON.parse(await toolPrompt.locator('#prompt-tools').textContent());
  check('tool-call Prompt shows the original input and offered tool',
    roundOne.at(-1).content === searchInput && toolSchema[0].function.name === 'web_search');
  await toolPrompt.goto(base + await page.locator('.message[data-message-role="assistant"] a').getAttribute('href'));
  const searchedRequest = JSON.parse(await toolPrompt.locator('#prompt-messages').textContent())
    .filter(message => message.role !== 'system');
  check('answer Prompt separates raw user input from paired tool results',
    searchedRequest[0].content === searchInput
    && searchedRequest.map(message => message.role).join() === 'user,assistant,tool,tool'
    && searchedRequest[2].tool_call_id === searchedRequest[1].tool_calls[0].id
    && searchedRequest[3].tool_call_id === searchedRequest[1].tool_calls[1].id);
  await toolPrompt.close();
  await page.screenshot({ path: path.join(artifacts, 'tool-search-timeline.png'), scale: 'css' });
  await page.locator('.message[data-message-role="user"] .fork-message').click();
  await settle(page);
  check('forking at user excludes later tool history and clears permission',
    await page.locator('.tool-event').count() === 0 && !await page.locator('[name="web_search"]').isChecked());
  await page.locator('.request-response').click();
  await finished(page);
  check('responding to the clean fork does not silently execute a search',
    await page.locator('.tool-event').count() === 0
    && await page.locator('.message[data-message-role="assistant"]').count() === 1);
  await page.goto(searchSourceURL);
  await page.locator('.tool-result .fork-event').first().click();
  await settle(page);
  check('forking through the first result excludes the second result and answer',
    await page.locator('.tool-call').count() === 1 && await page.locator('.tool-result').count() === 1
    && await page.locator('.message[data-message-role="assistant"]').count() === 0
    && await page.locator('.continue-response').isVisible());
  await page.locator('.input-tools-toggle').click();
  await page.locator('[name="web_search"]').check();
  await page.locator('.continue-response').click();
  await finished(page);
  check('explicit continuation reuses included results and completes only the missing call',
    await page.locator('.tool-call').count() === 1 && await page.locator('.tool-result').count() === 2
    && await page.locator('.message[data-message-role="assistant"]').count() === 1);
  await page.locator('[name="web_search"]').uncheck();
  await page.locator('.regenerate-response').click();
  await page.locator('.generation-progress').waitFor();
  check('regeneration retains the selected tool history while starting a clean attempt',
    await page.locator('.selected-tool-timeline .tool-result').count() === 2
    && await page.locator('.current-tool-timeline .tool-result').count() === 0);
  await finished(page);
  check('regeneration respects the current disabled search permission',
    await page.locator('.tool-event').count() === 0 && await page.locator('.variant-position').textContent() === '2/2');
  await page.getByRole('button', { name: 'Previous response', exact: true }).click();
  await settle(page);
  check('selecting a response version restores its own tool history',
    await page.locator('.tool-result').count() === 2 && await page.locator('.variant-position').textContent() === '1/2');
  await mobilePage.goto(page.url());
  check('tool history fits a 320px viewport', await fit(mobilePage));
  await mobilePage.screenshot({ path: path.join(artifacts, 'mobile-tool-history.png'), scale: 'css' });
  check('no browser console or JavaScript errors', errors.length === 0);
  console.log(`\n${checks.length} browser checks passed. Screenshots: ${artifacts}`);
} finally {
  await browser?.close();
  if (server.pid && server.exitCode === null) {
    server.kill('SIGTERM');
    await new Promise(resolve => server.once('exit', resolve));
  }
}
