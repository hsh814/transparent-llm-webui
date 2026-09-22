# Browser QA inventory

Status: browser verification completed on 2026-09-22 using Chromium, a temporary database, and simulated model responses.

The playwright-interactive skill is the intended workflow. This Codex installation reports its required `js_repl` feature as removed, so this run uses persistent Playwright handles in a Node REPL instead.

| Visible behavior / controls | Functional checks | Visual states and evidence |
|---|---|---|
| First use, New chat, keyboard shortcut | Create initial workspace; create a second chat; switch and return | Welcome and empty chat at desktop/mobile sizes |
| Chat composer, Enter, multiline input | Send with button and keyboard; preserve newlines; empty input rejected | Streaming and finished multiline reply; composer visible |
| Draft persistence | Type, navigate away/back, reload, browser back/forward | Restored draft and unchanged caret-independent content |
| Scroll behavior and Latest message | Read earlier output while stream continues; jump back | Long conversation, scroll position, jump control |
| Stop and Retry | Stop a slow response; retry; simulated failure then retry | Partial response, clear error, pending and completed states |
| Model selector, Refresh, Settings, Apply | Change model; change settings; save; reopen; invalid input | Settings panel on desktop and narrow mobile, toast |
| Folder create, fold, pin, settings, mode menu | Create/rename, expand/collapse, pin/unpin; prompt updates | Sidebar expanded/collapsed; menu layering; prompt visible |
| Search, Show all | Search titles/folders; clear query; expand >10 sessions | Filter results and restored sidebar state |
| Chat rename, delete; folder delete | Rename; cancel deletion; delete inactive and active items | Correct active chat, no stale composer |
| Copy message, Copy chat, Prompt page | Clipboard matches source; inspect exact submitted prompt | Prompt page, reasoning details, request settings |
| Memo mode | Save note; Enter inserts newline; no model request | Saved note and memo composer |
| Translation mode and chunk limit | Submit multiple chunks; paired output; stop/retry/reload batch | Dense chunked conversation with one stream connection |
| Mobile drawer, Escape, scrim | Open/close cycle, keyboard dismissal, select chat | 390×844 and 320×640, drawer layering and reachable controls |
| Layout / accessibility | Tab focus, horizontal overflow, main-region bounds | Screenshots of startup, dense state, settings, errors, mobile |
| Exploratory: fast generation and navigation | Send an immediate response, switch chats mid-stream, reload | No stuck pending messages or missing response |
| Exploratory: long content and literal HTML | Long title/prompt, code indentation, literal script tags | No script execution, clipping, or overlapping controls |

## Results

- 68 interactive assertions passed across the inventory above. Early assertion timing was adjusted to wait for htmx rendering before judging the visible state.
- A repeatable suite, `npm run test:browser`, passed 25 browser checks with no console or JavaScript errors. It creates its own preview server and database and cleans up its browser/server processes.
- Separate screenshot review covered welcome, conversation, open settings, partial/stopped output, prompt inspection, notes, translation pairs, long history, dense sidebar menus, and the mobile drawer. Viewports: 1440×900, 390×844, and 320×640.
- Bounds checks passed for the header, context bar, message area, and composer. Final screenshots show no horizontal overflow, obscured Apply/Delete controls, or clipped folder menus in the tested states. Browser Back restored the draft, scrolling stayed under user control, and switching away during generation did not duplicate replies.
- Exploratory checks included an immediate response, navigation/reload during a slow response, an aborted send, a failed generation followed by retry, folder mode round trips, a crowded sidebar, invalid settings, and literal HTML in model output.
- Live Ollama requests, non-Chromium engines, native phone keyboards, and the unavailable `js_repl` transport were not tested. Mobile checks used Chromium touch emulation; the exact playwright-interactive transport was replaced with a persistent Node REPL.
- Cleanup completed: both interactive browser contexts, the browser, the Node REPL, and the isolated preview server were closed. The user's chat database was not used.

## Bugs found and fixed

1. Stop inherited an htmx disabled-element selector that did not match its target. The Stop control now disables itself during its request.
2. htmx temporarily retained the previous `streaming` CSS class while replacing a same-ID bubble. That could change a stopped label back to “Responding” and keep Send disabled. Busy state now reads the new generation-status attribute, and controls are reconciled after settling.
3. Apply fell below the visible settings panel on a 320px screen with Advanced expanded. It now remains in a sticky panel header and submits the associated settings form.
4. A folder dropdown near the bottom of a crowded sidebar was clipped by the scrolling container. The menu now uses viewport-aware placement and closes cleanly on scrolling, resizing, or Escape, including resetting its expanded state.
5. Browser caching kept an older stylesheet after reloads during iteration. App CSS/JavaScript URLs now include the asset modification version so templates and assets stay in sync.

## Screenshot evidence

The repeatable run saved its reviewed artifacts under `test-results/browser-6Wplmi/`: `welcome.png`, `desktop.png`, `stopped.png`, `folder-menu.png`, `mobile-settings.png`, `mobile-drawer.png`, and `mobile-chat.png`.

Additional interactive screenshots are in `/private/tmp/transparent-chat-qa.Atk1oK/`, including `05-prompt.png`, `06-reading-history.png`, `07-memo.png`, `08-translation.png`, `12-small-settings-fixed.png`, `14-small-memo.png`, `15-dense-sidebar-menu-fixed.png`, and `16-mobile-folder-menu.png`. Screenshots ending in `fixed` replace the earlier failing state; intermediate drawer captures were taken during motion and were followed by settled captures.
