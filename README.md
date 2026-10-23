# Transparent Chat

A single-user workspace for conversations, translation, and notes, with image/PDF attachments and opt-in web search. FastAPI, htmx, and SQLite; no frontend build step or JavaScript framework. Model requests go to Ollama Cloud.

## Setup and run

```bash
uv sync
```

Create a `.env` file in the project root:

```dotenv
OLLAMA_API_KEY=your-key
```

```bash
uv run uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000`. An API key is required at startup. Run **one application worker**: generation workers and restart recovery are coordinated within a single process. This is a local, single-user application without authentication; do not expose it publicly.

## Using the workspace

- **New chat** starts a conversation in the current folder, creating a Personal folder on first use. `⌘/Ctrl K` does the same.
- Search the sidebar to find a chat or folder. Pin frequently used folders using the star.
- Titles use the first message or note, with whitespace normalized and a 60-character limit (including an ellipsis when truncated). No model call is needed. Empty chats get unique numbered labels. Click a title to rename it; changes save on blur or Enter. Clear it to restore the automatic title. Existing unnamed chats are backfilled on startup without changing their modification dates.
- Each sidebar chat shows its last-modified date and time in your browser's timezone; hover the date for the full timestamp. Sending, receiving, renaming, changing settings, stopping/retrying, and deleting messages refresh it.
- Use the folder menu to choose **Chat**, **Memo**, or **Translation**. Folder settings contain the name and system prompt, plus the translation chunk size.
- The current model stays in the header. Open **Settings** to adjust reasoning effort, temperature, top P, maximum output tokens, or the random seed; click **Apply** to save.
- A configured system prompt is shown above the conversation. It is sent verbatim. Empty prompts add no system message.
- On a desktop, Enter sends a chat message and Shift + Enter inserts a newline. Notes and translations use Enter for newlines and `⌘/Ctrl Enter` to submit. On touch devices, Enter inserts a newline.
- Text drafts are stored per chat in browser local storage and cleared only after a successful submission. They survive navigation and reloads, including failed sends. Unsubmitted attachments stay in browser memory across chat navigation and failed sends, but are lost on reload; remove them individually before sending if needed.
- **Stop** stops accepting response output and cancels queued translation chunks. Partial output remains visible. **Retry with saved settings** resumes the latest unfinished submission, retaining completed tool results and retrying only unresolved calls or the unfinished model round. The current **Allow web search** checkbox controls permission to execute pending searches. A stopped upstream connection closes when the worker next receives data or times out.
- Reading earlier messages does not force-scroll to the bottom. **Latest message** returns to the newest output. The empty text box is one line tall and grows with its content. Its action toolbar sits outside, directly below the text box, and stays visible on mobile.
- **Prompt** is available on user messages, unfinished responses, assistant tool calls, and final answers. Each tool-call Prompt shows the input to that model round; the final answer's Prompt includes the preceding assistant calls and separate tool results. It preserves text whitespace and previews image parts. **Exact messages JSON** includes full image base64; **Copy messages JSON** copies it without headings or the response. Model, saved settings, and the offered tools schema are inspectable separately. A saved or queued request is not proof of upstream delivery. **Copy chat** copies user/answer text, not attachment bytes or tool history.
- Links to `/?session=<id>` open a complete page and survive reloads and browser navigation.

### Forks and response versions

- **Fork** on a saved message creates an independent chat in the same folder, copying the visible conversation **through that message**. It copies only selected answers, not hidden alternatives or later turns. Message text, reasoning, and usage are copied; image/PDF inputs, saved requests, and search results use shared immutable references rather than duplicating their data. The new chat inherits the current model/settings and gets a distinct `· fork #<id>` title. Forking does not reprocess files, repeat searches, or request a model response; deleting the source chat does not delete its fork.
- Forking at a **user message** copies its original text and attachments but excludes every subsequent tool call, search result, and answer for that turn. Earlier selected turns retain their explicit history. **Request response** answers the existing user message without duplicating it, including after deleting its only answer.
- **Fork through call** and **Fork through result** stop at that exact tool event. Later calls/results and the final answer are excluded. A fork through the first result of a multi-call batch includes only that result, not the remaining results. Event forks stay idle; **Continue response** explicitly resumes them. Search permission defaults off in a new fork. If calls remain unresolved, enable permission to continue, or fork before the call; new user input cannot skip an unresolved idle call.
- **Regenerate** keeps the selected answer and its tool history visible while producing a new version from the saved base input. The new attempt starts with no searches of its own; the model may choose fresh searches if currently allowed. Use **← / →** and `1/3` to select an answer together with its own tool history. Only that selected version enters subsequent conversation history. Failure or Stop leaves the previous answer intact.
- Once a turn has follow-up messages, its version controls are locked. Fork at that answer first to change it without making the existing follow-up history inconsistent.
- Requesting again and regenerating reuse saved input/model/parameters, not current model settings; **search permission always follows the current checkbox**. Retry/Continue reuse explicitly included completed results, whereas regeneration does not inherit the regenerated answer's searches. Legacy turns without a saved request use current settings; their original answers remain labeled as reconstructed in Prompt. Memo conversations support Fork but never model-response actions.
- **Prompt** on each response version shows its immutable saved request. Token usage includes all model rounds in that response; session totals include all retained response versions. Deleting the final selected answer selects the nearest remaining version and its trace; deleting the only version clears that turn's trace and leaves the original user input unanswered. Message mutations and forks are blocked while generation is active.

### Batch editing into a fork

- **Edit messages** opens an optional conversation editor; **Edit** on a message focuses that row. Edit several user inputs, assistant answers, and assistant **Thinking** fields together, exclude individual messages, and preview the resulting order. Thinking can be added, changed, or cleared independently of the answer. Only visible selected answers are included, not hidden response versions. Ordinary later messages remain included unless you exclude them.
- **Save as fork** atomically saves an independent conversation without making a model request. **Save & send** is available only when the last included message is a user input; it answers that saved input without adding a duplicate. Assistant-ending forks wait for your next user message. Empty forks can be saved, but cannot be sent. Memo editing only saves notes.
- The source conversation and its unsent composer draft stay unchanged. **Cancel** or Escape discards the editor changes. Editor drafts exist only in the open dialog until saved; they are not autosaved across reloads.
- Existing image/PDF attachments can be kept or removed per message. Kept attachments reuse their immutable data without uploading or rendering them again. To add new files, use the normal composer in the saved fork.
- Tool calls, arguments, and results are read-only; their assistant **Thinking** is editable. **Edit Thinking** on a tool request focuses its batch. A selection keeps a complete assistant tool-call batch together with every matching result, including recorded failures. Incomplete batches cannot be edited or included, and excluding the originating user excludes its tool history. From the first edited or excluded context item onward, tool batches default to **excluded**; explicitly select any historical results you want to reuse. Changing a batch's Thinking requires explicitly selecting that batch again. Reuse does not rerun searches, retry historical failures, or count their original model usage as new work.
- **Allow fresh web search for Send** is a separate, off-by-default permission. It allows the model to request new searches; it does not control whether explicitly selected historical results enter the edited context. The normal search-permission indicator reflects this choice after Save & send.
- Edited inputs, **User edited response**, and **Carried from earlier context** badges distinguish corrections from generated answers. Editing an assistant's answer or Thinking clears its current model and token attribution, but preserves unchanged Thinking. Manually edited Thinking is labeled separately. The source viewer preserves Thinking before the latest edit and the original generated Thinking; repeated forks do not turn manual work back into model-generated output. **Prompt** still shows that answer's actual historical request when known, never a fabricated request for the edited text; unknown requests remain explicitly unknown.
- New requests and regeneration use the fork's rebuilt edited context and the model, parameters, and system prompt captured when saving the fork. Historical answer Prompts remain unchanged. An assistant kept without its originating user is context only, not an answer that can be regenerated against an unrelated user.
- Editing is blocked while the source is generating. If its messages, selected versions, settings, or folder context change in another tab, saving rejects the stale snapshot without creating a partial fork. The dialog keeps your edits and displays the error; reopen against the updated source before saving.

### Images, PDFs, and web search

These controls are available in **Chat** folders; Memo and Translation remain text-only. **More inputs** and **Send** share a toolbar outside the one-line text box. Open **More inputs** to reveal attachment and search-permission controls. The panel starts collapsed whenever a chat is opened. When collapsed, the toggle shows attached-file counts and **Search allowed** if enabled; collapsing it does not remove files or revoke permission.

- **Attach images or PDF** accepts PNG, JPEG, static WebP, and PDF. A message may contain files without accompanying text. The selected model must support vision; the app never silently substitutes text-only input.
- Images are validated locally and sent as their original bytes in base64 image content parts.
- PDFs are processed locally with PDFium. **Every page is sent as a rendered image plus extracted text**, in page order. This preserves visual access to tables, figures, equations, and multi-column layouts rather than relying on text extraction alone. Scanned/textless pages are included as images; there is no local OCR. Renderings use JPEG quality 90 at up to 144 DPI, capped at 2,000 pixels on the long edge. These are page renditions, not the original PDF file.
- Expand a saved attachment to inspect its name, original byte size, SHA-256, page images, and extracted text. App-added labels and document-handling instructions are visible in Prompt alongside the exact image content. PDF extraction can lose reading order; vision models can still misread small labels, equations, or complex tables. Verify important values against the original paper; attach a close-up image when needed.
- **Allow web search** is off in a new chat or fork. It offers a `web_search` function to the model through [Ollama's OpenAI-compatible tool-calling API](https://docs.ollama.com/api/openai-compatibility). **The model decides whether to search and chooses the query**; checking the box does not search automatically or copy your input into a query. Use a model supporting [tool calling](https://docs.ollama.com/capabilities/tool-calling). There is no app-side query override and no `tool_choice` parameter.
- When the model requests a search, the worker calls [Ollama's `/api/web_search`](https://docs.ollama.com/capabilities/web-search) with the existing API key. Each response has a budget of **eight tool-call attempts**, including invalid-argument attempts. Every offered tool description states the remaining budget and tells the model to stop once it has enough evidence. Both the function and parameter descriptions state that `max_results` must be an integer from **1 to 5**, defaults to **5** when omitted, and cannot exceed **5**; the JSON schema carries the same limits. The app does not fetch full source pages.
- Invalid search fields (such as `max_results: 10`, a blank query, or extra arguments) are **not executed**. The timeline records a `rejected` result with the specific validation error, and the model receives it as a separate tool reply so it can correct its call or answer without searching. Original arguments stay unchanged; the app never clamps a request to five silently. A rejection is a resolved tool outcome, not a failed network request. Unsupported tools, malformed/incomplete tool-call envelopes, missing permission, and actual search API failures still stop visibly.
- **Budget exhaustion does not discard the answer.** Calls in a batch are handled in their recorded order: if six attempts have been used and the model requests three more, only the first two can execute; the ninth receives a separate `search_budget_exhausted` result with `executed: false`. Once all pending calls are resolved, the next model request offers **no tools** and includes an app-authored system instruction to answer from saved evidence, stating uncertainty when necessary. This also happens after exactly eight calls, without waiting for a ninth. If the provider still emits tool calls in that tools-free final round, they are recorded as unexecuted and the response stops explicitly instead of looping indefinitely.
- The conversation shows **user input → assistant tool request → tool result → answer** in actual order. Queries, arguments, status, retrieval time, source links, and expandable snippets appear at their own positions, not in an answer footer. Each tool round displays its saved budget; exhausted budget awaiting pending outcomes is distinguished from final-answer mode. **Prompt** shows the exact remaining budget, offered tools, and final-answer instruction. Search permission is saved separately from the tools currently offered: budget exhaustion does not revoke the checkbox permission. Input stays unchanged, and the one-response final-answer instruction does not enter subsequent turns. The model receives assistant `tool_calls` and separate `role: tool` messages paired by `tool_call_id`; source snippets are untrusted reference material, and only HTTP(S) links are clickable.
- Retry reuses completed results and resolved rejections, retrying only failed/unresolved calls or the unfinished model round. Earlier invalid-argument and hard-limit failures can resume through this flow; previously completed searches are not repeated, and only remaining budgeted searches can run. Retry after final-answer failure retains zero budget and requests another tools-free answer. Fresh searches still require permission. Regeneration starts a new response budget; event forks retain the prefix's consumed budget, while complete explicitly retained outcomes in an edited fork are context rather than new attempts. Excluding a retained result makes its reexecuted call count again. Failed/cancelled tool outcomes remain visible and are paired error replies in later conversation history. To exclude a turn's search context, fork at its user message rather than after a result.

| Per-submission limit | Value |
|---|---|
| Files | 4 |
| Original file size | 10 MiB each, 20 MiB combined |
| Image dimensions | 20 million pixels per image |
| PDF pages | 20 combined across all PDFs |
| Extracted PDF text | 100,000 characters combined |
| Rendered PDF image bytes | 32 MiB combined |

Invalid, unreadable, password-required, or over-limit attachments reject the entire submission before any search or model request. Pages/text are never silently truncated; split longer documents yourself. The selected model's context/image limits may be lower and can still cause an upstream rejection.

**Privacy and storage:** the model sees the conversation and attachments and may derive a search query from any of them. Only that generated query and result limit go to the search endpoint. Original images, PDF page renditions, extracted text, and tool results go to the configured model service. SQLite stores inputs and request/trace snapshots as shared immutable, content-addressed JSON. Files and unchanged serialized tool results are reused across forks, retries, and retained versions; fresh searches create new snapshots without changing another branch. Deduplication is local only: API requests still contain complete message/image parts. Original PDF files are not retained; keep them separately. Local storage and the database are not encrypted by this application.

Deleting a message, chat, or folder removes shared data only when no remaining message, saved request, or edit-provenance record references it. A later request or historical provenance can therefore retain inputs from an earlier deleted message. Removing an attachment from edited context does not erase it from the original Prompt or provenance. Unreferenced payload rows are reclaimed transactionally; SQLite may retain freed pages for reuse, so the database file does not necessarily shrink immediately. Startup does not automatically run `VACUUM`.

### Folder modes

**Chat** sends the system prompt plus the visible conversation history, including only the selected response version for each turn. A second submission is rejected while generation is in progress, including requests from another tab.

**Memo** saves notes without model calls, including when opening the chat. Converting a memo folder preserves its previous system prompt for later use.

**Translation** splits the input at line or sentence boundaries, with a hard character limit as a fallback. Splitting preserves the source text. Each chunk is translated independently with the system prompt and that chunk only. Source and translated text appear together, in order. The batch continues if the browser closes; failures pause the remaining chunks until retry.

## Reliable generation and prompt history

The send transaction saves the raw user input and base request: model, supported parameters, system prompt, attachments, ordered API messages, and allowed tools. The worker first calls the model. Each returned tool-call message is saved with that round's exact input and usage, then tool outcomes are persisted separately before the next model round. Current request, base request, and ordered trace use immutable shared roots; response versions own their snapshots. Search never modifies user content or the folder's system prompt.

A background worker generates each response once. Generation state, partial output, reasoning, and errors are stored in SQLite. SSE connections display this state; they never initiate duplicate model calls. There is at most one live SSE connection per conversation view, including translation batches. Escaped HTML and multiline SSE framing preserve line breaks, indentation, and literal HTML in model output.

On reconnection, stored content replaces the display instead of being appended twice; tool-round transitions also clear obsolete partial output. Completed answers retain their request and tool trace, with usage summed across model rounds. Regeneration creates another version. Restarted work is marked interrupted, including unresolved tool outcomes, and can be explicitly retried. Attempt counters prevent late model/search results from overwriting a newer attempt; selecting an older version never rewinds the counter.

Older messages remain available. Unknown original requests are labeled as reconstructions. Startup transactionally migrates inline payloads and old search preparation into shared request/base/trace roots. Old automatic searches are labeled **Legacy app search — not requested by the LLM** in the timeline. Historical Prompt messages remain exactly as originally saved, including any old inline search injection; future history and user-boundary forks use clean user text/attachments and separate tool messages. Migration preserves IDs, dates, selected versions, and historical snapshots, and rolls back on failure. Back up the database before upgrading; migration can temporarily require additional disk space.

### Thinking in request history

- Nonempty assistant Thinking is sent in the separate **`reasoning`** message field supported by Ollama's `/v1/chat/completions`. It is included both in subsequent Chat turns and when continuing after a tool result. It is never appended to user input or answer `content`. The selected response version and explicitly retained tool history determine which Thinking is sent. Translation still sends only its own source chunk, not previous answers.
- The worker accepts streamed `reasoning` and `reasoning_content` output and stores Thinking separately. Outgoing Ollama requests use `reasoning`; this is not a claim that every OpenAI-compatible service accepts that field. See [Ollama's tool-loop guidance](https://docs.ollama.com/capabilities/tool-calling#streaming) and [OpenAI-compatibility conversion](https://github.com/ollama/ollama/blob/main/openai/openai.go).
- **Prompt** shows **Thinking included in this request** separately from answer content, or explicitly indicates its absence. Exact messages JSON contains the same field sent by the HTTP adapter. Displaying saved Thinking in the conversation does not mean an older request contained it; server/model templates can also decide whether to consume a transmitted field.
- Original historical requests are never rewritten. When explicitly retrying, regenerating, or forking an older saved request that omitted Thinking, the app adds it to the new request only when the matching saved turn and response identify it unambiguously. Conflicting identical response variants or missing source records are not guessed. An edited fork rebuilds its context from the selected visible messages and Thinking.
- Thinking now contributes to the request payload and can increase context usage. It is sent to the configured model service and can inform model-generated search queries. Editing or clearing it only changes the new fork's context; the source and provenance retain the original records. Existing `reasoning` storage is reused without a database-column migration.

### Supported model settings

The app uses Ollama's `/v1/chat/completions` endpoint. Seed is sent as a top-level field. Context size, top K, and repeat penalty are not exposed as controls because this endpoint does not document support for those fields; old stored values are retained but omitted from new requests. See [Ollama's supported request fields](https://docs.ollama.com/api/openai-compatibility).

Context capacity is informational when it can be retrieved. Reasoning-level support depends on the model. The selected model remains available in the selector even if it disappears from the catalog. Metadata requests use short timeouts; successful lookups are cached for five minutes and failures for 30 seconds. The refresh button clears the cache.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_API_KEY` | Required | Ollama Cloud bearer token |
| `OLLAMA_API_BASE_URL` | `https://ollama.com` | Native model metadata and web-search API |
| `OLLAMA_OPENAI_BASE_URL` | `https://ollama.com/v1` | Model list and chat completions |
| `CHAT_DB_PATH` | `chat.db` beside `db.py` | Database path; useful for isolated development |

Back up `chat.db` while the application is stopped to preserve your conversations and request history.

## Development and verification

```bash
uv run python -m unittest discover -s tests -v
node --check static/app.js
```

Tests use temporary databases and mocked Ollama responses; they do not contact live services or modify your chat database. Coverage includes streamed multi-call argument assembly, model-chosen queries, permission without forced search, distinct roles, multi-round limits, tool failure/cancellation/restart, completed-result reuse, per-round Prompt fidelity, event-boundary forks, stale trace/attempt rejection, response-version traces, and legacy migration rollback. Existing image/PDF fidelity, payload sharing/reclamation, translation, memo, concurrency, settings, and deletion regressions remain covered.

### Browser checks

Playwright is an optional development dependency; the application still needs no frontend build.

```bash
npm install
npx playwright install chromium
npm run test:browser
```

The browser suite starts and stops an isolated preview server with simulated model/tool responses. It exercises real controls, streaming, drafts, settings, fork/version/deletion workflows, inline tool progress and snippets, per-round Prompt, user/result boundary forks, continuation without duplicate results, current search permission, and layouts at 320px and 390px. Screenshots go under ignored `test-results/`. On macOS it can use `/Applications/Chromium.app`; `PLAYWRIGHT_CHROMIUM_EXECUTABLE` overrides that choice. Use `HEADED=1 npm run test:browser` to watch.

The interactive coverage inventory and results are in [tests/UI_QA.md](tests/UI_QA.md).

```text
app.py           App setup, startup recovery, full-page navigation
db.py            SQLite storage and atomic generation state transitions
payloads.py      Shared immutable JSON nodes, content hashes, reference-aware reclamation
generation.py    Background model/tool loop, durable continuation, safe errors
tool_events.py   Search tool schema, ordered API projection, aggregate usage
ollama.py        API payloads, web search, response parsing, metadata cache
inputs.py        Validated images, PDF page rendering/text extraction, content parts
templating.py    Shared template helpers and ordered conversation rendering
routers/         Folder, session, chat, fork, response-version, and prompt-viewer routes
templates/       Jinja HTML and htmx/SSE fragments
static/app.js    Drafts, keyboard actions, scroll behavior, feedback, sidebar
static/app.css   Responsive workspace styles
tests/           Offline regression suite
```

The principal routes remain HTML/SSE endpoints: `/folders`, `/folders/{id}/sessions`, `/sessions/{id}`, `/sessions/{id}/send`, `/sessions/{id}/stream?since=<user_message_id>`, `/sessions/{id}/memo`, `/sessions/{id}/messages`, and `/sessions/{id}/messages/{id}/prompt`. `/start` creates a chat, `/sessions/{id}/stop` cancels unfinished work, and `/sessions/{id}/retry` retries the latest failed or stopped submission. Message actions use POST `/sessions/{id}/messages/{id}/fork`, `/respond`, and `/regenerate`; version selection uses POST `/sessions/{id}/messages/{user_id}/variants/{assistant_id}/select`. Only the final **visible** message can be deleted, even when its selected response has an older database ID. Message mutations are blocked while work is active.

Tool events add POST `/sessions/{id}/messages/{user_id}/events/{index}/fork` with a required `trace_id` form value, and GET `/sessions/{id}/messages/{user_id}/events/{index}/prompt?trace_id=...` for assistant-call rounds. Trace hashes reject stale boundaries instead of silently forking or inspecting another response version.
