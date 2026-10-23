# Browser QA inventory

## Search budget finalization

- Verified 103 offline regressions. Budget coverage includes 3+3+3 requests executing only the first eight, exact-eight exhaustion, mixed valid/invalid attempts, saved hard-limit failures and final-model failures on Retry, permissions and unsupported-tool preflight, event-fork prefixes, edited retained results, regeneration reset, and finite failure when a provider ignores the tools-free final-answer instruction.
- A throwaway `httpx.MockTransport` smoke exercised the production HTTP adapters and worker with remaining budgets 8, 5, 2, then 0: four model requests, eight search requests, and one paired `search_budget_exhausted` result with `executed: false`. The last model request omitted tools, included the explicit final-answer system instruction and saved Thinking, and completed an answer. Each saved Prompt matched the actual outgoing messages/tools. The finalization instruction did not leak into the next user turn.
- Interactive Chromium exercised the same 3+3+3 workflow through the real UI and isolated preview provider. The timeline showed each round's budget, eight completed searches, the ninth unexecuted call, final-answer mode, and a completed answer. The search-permission checkbox stayed enabled independently of exhaustion. The exact Prompt showed zero budget, enabled permission, no offered tools, and the app-authored instruction.
- Desktop and 320px views showed the budget outcome and final-answer notice without horizontal overflow; browser errors were empty. Reviewed artifacts: `test-results/search-budget-aPUqR6/budget-desktop.png`, `budget-mobile.png`, and `budget-prompt.png`. No real Ollama request or user-database mutation was made.

## Search argument recovery

- Verified 97 offline regressions. New cases cover two invalid `max_results: 10` calls corrected to five, invalid query/types/extra fields, mixed valid/invalid batches, the omitted-result default, old failed-call Retry, continuation failures, fine-grained and edited forks, stale attempts, permission boundaries, and the eight-call limit including rejections.
- A throwaway `httpx.MockTransport` smoke exercised the production model/search HTTP adapters and worker: three model requests, exactly one search request with `max_results: 5`, the original `10` preserved, and a separate paired `invalid_arguments` tool reply with `executed: false`. Saved round Prompts matched outgoing messages, including Thinking. No real Ollama request was made.
- Interactive Chromium exercised rejection → corrected call → saved source → answer with no generation failure. The correction round's Prompt contained the unchanged original call, its Thinking, and the validation tool reply. Forking through the rejection and continuing with search permission off produced an answer without rerunning the rejected call or including the later search.
- Desktop and 320px screenshots show the rejected search in its chronological position, explicitly marked not executed, with no horizontal overflow. Artifacts: `test-results/search-recovery-eBJcqM/validation-desktop.png` and `validation-mobile.png`. Browser errors were empty. All preview sessions used a temporary database; the user's database was untouched.

## Thinking replay and editing

- Verified 89 offline regressions and 79 repeatable browser checks. Sources, forks, request snapshots, and test servers used isolated databases; no real Ollama request was made.
- Worker regressions exercise Thinking across multiple search rounds and the next user turn, selected-version/fork context, retry and event-fork continuation, Thinking-only edits and answers, clearing, repeated provenance after source deletion, downstream tool invalidation, unchanged call/result pairs, and atomic rejection of invalid edits.
- Older saved bases gain recoverable Thinking only in newly requested work. Tests preserve the exact old Prompt and original model/settings, and reject guessing between identical answers with conflicting Thinking.
- A throwaway `httpx.MockTransport` smoke exercised the production HTTP adapter and worker: three model HTTP requests, separate `reasoning` in the tool continuation, both edited tool and final Thinking in the next turn, and saved Prompt exactly matching the outgoing messages. Incoming `reasoning_content` was captured and replayed using Ollama's outgoing `reasoning` field.
- Interactive Chromium used a two-round search conversation. Editing the first tool's Thinking excluded both affected batches until the first was explicitly reselected; editing the final Thinking in the same fork retained its unchanged answer text. The next request included only that selected batch and both edited Thinking fields. Source inspection showed current, immediate-before, and original generated Thinking separately.
- At 320px, the Thinking field and sticky save controls were visible together. Clearing the final answer's Thinking and using Save & send removed that field from the outgoing assistant message while retaining the selected tool's Thinking, with no duplicate user input. Browser errors were empty.
- Reviewed screenshots: `test-results/browser-dZHcUb/batch-editor-desktop.png`, `batch-editor-mobile.png`, `thinking-editor-desktop.png`, and `thinking-editor-mobile.png`. The repeated suite also verifies HTML/whitespace preservation without script execution, exact historical Prompt, and manual attribution without fabricated usage.

## Batch-edit forks

- Verified 80 offline regressions and 74 repeatable browser checks with isolated databases and simulated model/search responses. Reviewed desktop and 320px editor screenshots in `test-results/browser-kMNSsv/`.
- Backend coverage includes multiple user/assistant corrections, message and attachment exclusions, manual/carried provenance, actual historical Prompt preservation, rebuilt regeneration context, shared payload lifetime after source deletion, stale snapshots, active-source rejection, atomic Save & send, empty and assistant-ending forks, and Memo/Translation request boundaries.
- Tool-history checks cover explicit selection after context changes, atomic multi-call batches, incomplete/orphan history rejection, retained failed results without retry, and exclusion of historical usage. An event fork through a retained call reproduced nine searches against an eight-call limit; replayed calls now count toward the limit, while reused results do not.
- Browser workflows cover Cancel/Escape without altering the source or composer draft, editing both roles in one save, manual-response attribution, the original answer's exact Prompt, edited context in a subsequent request, explicit reuse of saved search results, Save & send without duplicate input, fresh-search permission, and disabled editing during generation.
- Interactive Chromium smoke uploaded a PNG and a real two-page PDF, removed only the PNG in the editor, and used Save & send. The actual saved Prompt retained both PDF page images and extracted text, without the removed image or duplicate input.
- A second tab changed the source while an editor draft was open. Saving returned 409, created no fork, and kept the correction in the dialog. The initial error appeared below the visible scroll area; error handling now brings it into view above the sticky actions. Confirmed visible errors and reachable controls at desktop and 320px; screenshots: `batch-editor-conflict-desktop.png` and `batch-editor-conflict-mobile.png`.
- An offline save showed an inline connection error, preserved the open draft, and re-enabled Save. Expected conflict/offline failures were intentional; the repeatable browser run had no console or JavaScript errors. Editor drafts are not persisted across reloads.
- The compact main composer remains unchanged. Testing used only isolated preview databases and servers; no live Ollama request or user-database mutation was performed.

## Model-driven search and event forks

- Verified 67 offline regressions and 59 repeatable browser checks with isolated databases and simulated model/search responses. Browser artifacts: `test-results/browser-cMJ00n/`.
- Coverage includes streamed multi-call fragments, permission without forced search, model-selected queries, separate assistant/tool roles, per-round Prompt, multi-call result boundaries, explicit continuation, variant-owned traces, cancellation/restart pairing, stale writes, and legacy migration without losing historical inputs.
- A throwaway wire-level smoke exercised the production HTTP adapters and worker using `httpx.MockTransport`: two model requests, one generated-query search, exact per-round message/tool snapshots, unchanged user input, and aggregate usage of 22 tokens. No `tool_choice` was sent.
- Interactive Chromium smoke uploaded an original PNG and a real two-page PDF, completed two search rounds, and inspected the second round's Prompt: user → assistant call → tool result, with all three image parts preserved and no search text injected into user content. Forking after the first result excluded the second round; Continue with permission off reused that result without another search.
- Source snippets were read inline; running status appeared before the answer. Desktop and 320px surfaces were visually inspected, with unique DOM IDs, no horizontal overflow, and no browser errors. Corrected assistant-call card padding so generic assistant styles do not remove its inset.
- The two interactive chats retained three shared image values rather than duplicating their PNG/PDF page bytes. The test database, browser tabs, and preview server were isolated from the user's workspace data. Live Ollama behavior remains dependent on the selected model's tool support and was not exercised against the cloud.
- A rejected-search state probe caught a transient `running` event even though permission prevented any network call. Rejections now persist failed outcomes directly; the same probe confirmed zero intermediate running states and zero search requests.

## Shared immutable payload follow-up

- Verified 60 offline regressions and 46 repeatable browser checks. Migration tests preserve historical variants, explicit unknown legacy requests, IDs/dates, unfinished attempts, and shared references; an injected mid-migration failure rolls back and permits retry.
- Interactive Chromium smoke submitted one image, a two-page PDF, and web search, then regenerated and forked the selected answer. Stored payload data remained 67 nodes / 29,904 body bytes with three image values; all three saved responses shared one request root.
- After deleting the source chat, the fork's Prompt JSON matched exactly and its PDF page still rendered. Deleting the final fork left zero payload nodes and edges. Browser errors were empty; real Ollama was not contacted.
- A separate throwaway legacy database reproduced the old duplication, migrated successfully, and retained identical requests after source deletion. An additional fork added zero immutable payload bytes.
- Repeatable browser artifacts: `test-results/browser-aAxmqP/`. Shared-file preview was visually inspected in a separate isolated browser session. The user's database was not used.

## Fork / response-version follow-up

- Verified 56 offline regressions and 46 repeatable browser checks with temporary databases and simulated model/search responses.
- Browser coverage includes retained answers during regeneration, repeated previous/next selection without reload, selected-version follow-up context, intermediate-message forks, deletion followed by Request response without duplicate input, and working response controls at 320px.
- Additional interactive checks used an image and a two-page PDF with search: the following request retained all three image parts and only the selected answer; forks preserved the selected prefix. Forking at a user message stayed idle until explicitly requested. Stopping regeneration preserved the complete prior answer, and retry added a new version.
- Fixed a real htmx lifecycle bug: replacing a mutation button retargets `afterRequest` to a surviving ancestor. Tracking the request's XHR rather than the removed button releases the busy lock reliably. The browser suite exercises consecutive mutations to prevent recurrence.
- Desktop and settled 320px screenshots were inspected. The text box remains one empty line, with More inputs and Send outside it; response actions fit without horizontal overflow. The repeatable run saved screenshots in `test-results/browser-unI23B/`, including `mobile-response-versions.png`.
- No live Ollama call was used. Mobile verification used Chromium touch emulation; a cancelled SSE connection during Stop is expected.

## Automatic title / last-modified follow-up

- Requirements: unique empty-chat labels; deterministic first-input titles, normalized whitespace and 60-character truncation; preserve manual titles; clearing a title restores automatic mode; backfill legacy unnamed chats without changing their dates.
- Controls/states: create, send, rename, clear, search, reload; timestamps must reconcile after completed/stopped/retried responses, note saves, deletions, and settings changes. Backend tests cover migration and tiny translation chunks.
- Browser checks: title in header/sidebar agrees, later messages do not replace it, manual/automatic round trip persists, UTC timestamps display in the browser timezone and refresh after completion. Inspect desktop long-title state and 320px drawer screenshots for visible dates and no overlap/overflow.
- Exploratory cases: a cleared title and Unicode/multiline content; filtering by title after adding date labels. Keep all tests on an isolated database with simulated replies.
- Results: 33 backend tests and 35 repeatable browser checks passed. A persistent headed Node/Playwright session additionally exercised Unicode/multiline input, manual rename, clearing with Enter, numbered empty chats, title search, and the 320px drawer. The Enter check found and fixed a focused header input staying blank after the sidebar had restored its automatic title.
- Reviewed follow-up screenshots: `test-results/browser-DFW4dw/titles-desktop.png`, `test-results/browser-DFW4dw/mobile-drawer.png`, and `/private/tmp/transparent-titles.qDRjUM/mobile-titles.png`. Titles and local dates remain legible without overlap; viewport checks pass. Browser/server cleanup completed; tests did not use the user's database. Legacy backfill takes effect on application restart.

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
