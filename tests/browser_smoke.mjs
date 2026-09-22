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
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
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
  check('CSS and JavaScript URLs are versioned', await page.locator('link[rel=stylesheet]').getAttribute('href').then(href => href.includes('?v=')));

  await page.locator('.composer textarea').fill('[slow] Browser stop regression');
  await page.locator('.composer textarea').press('Enter');
  await page.locator('.streaming .content span').waitFor();
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
  await page.getByRole('button', { name: 'Retry with original settings' }).last().click();
  await finished(page);
  check('retry finishes without duplicating the response', await page.locator('.message.assistant:not([data-generation-status])').count() === 1);
  check('literal model HTML remains inert', await page.evaluate(() => !window.unsafeExecuted));
  check('output indentation is preserved', (await page.locator('.message.assistant:not([data-generation-status]) .content').textContent()).includes('\n    indented  code'));

  const draft = 'Draft survives\npage changes';
  await page.locator('.composer textarea').fill(draft);
  await page.locator('.new-conversation').click();
  await settle(page);
  await page.locator(`.session-link[hx-get="/sessions/${sessionId}"]`).click();
  await settle(page);
  check('draft survives navigation', await page.locator('.composer textarea').inputValue() === draft);
  await page.reload();
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
  check('no browser console or JavaScript errors', errors.length === 0);
  console.log(`\n${checks.length} browser checks passed. Screenshots: ${artifacts}`);
} finally {
  await browser?.close();
  if (server.pid && server.exitCode === null) {
    server.kill('SIGTERM');
    await new Promise(resolve => server.once('exit', resolve));
  }
}
