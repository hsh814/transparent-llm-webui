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
- **Stop** stops accepting response output and cancels queued translation chunks. Partial output remains visible. **Retry with original settings** retries the latest unfinished submission using its stored request. A stopped upstream connection closes when the worker next receives data or times out.
- Reading earlier messages does not force-scroll to the bottom. **Latest message** returns to the newest output. The empty text box is one line tall and grows with its content. Its action toolbar sits outside, directly below the text box, and stays visible on mobile.
- **Prompt** is available on new user messages, unfinished responses, and assistant messages. It shows saved API messages in order, preserving text whitespace and previewing image parts. **Exact messages JSON** reveals the complete messages array, including image base64; **Copy messages JSON** copies it without headings or the response. Model, saved settings, attachments, and search provenance are shown separately. Queued or search-incomplete requests are explicitly labeled; a saved request is not proof of upstream delivery. **Copy chat** copies displayed message text, not attachment bytes or provenance.
- Links to `/?session=<id>` open a complete page and survive reloads and browser navigation.

### Images, PDFs, and web search

These controls are available in **Chat** folders; Memo and Translation remain text-only. **More inputs** and **Send** share a toolbar outside the text box. Open **More inputs** to reveal attachment and web-search controls between the text box and toolbar. The panel starts collapsed whenever a chat is opened. When collapsed, the toggle still shows the number of attached files and **Search on** if enabled; collapsing the panel does not remove files or turn search off. Detailed limits and processing behavior are documented below rather than in a composer help disclosure.

- **Attach images or PDF** accepts PNG, JPEG, static WebP, and PDF. A message may contain files without accompanying text. The selected model must support vision; the app never silently substitutes text-only input.
- Images are validated locally and sent as their original bytes in base64 image content parts.
- PDFs are processed locally with PDFium. **Every page is sent as a rendered image plus extracted text**, in page order. This preserves visual access to tables, figures, equations, and multi-column layouts rather than relying on text extraction alone. Scanned/textless pages are included as images; there is no local OCR. Renderings use JPEG quality 90 at up to 144 DPI, capped at 2,000 pixels on the long edge. These are page renditions, not the original PDF file.
- Expand a saved attachment to inspect its name, original byte size, SHA-256, page images, and extracted text. App-added labels and document-handling instructions are visible in Prompt alongside the exact image content. PDF extraction can lose reading order; vision models can still misread small labels, equations, or complex tables. Verify important values against the original paper; attach a close-up image when needed.
- **Web search** is off in a new chat. Enabling it sends your message, or the optional explicit query override, to [Ollama's `/api/web_search`](https://docs.ollama.com/capabilities/web-search), using the existing API key. Files and extracted PDF text are not automatically used as the search query. The choice and query are remembered per chat. Each submission makes one search for up to five results; this is not an autonomous browsing loop and does not fetch full source pages.
- Search status, query, result limit, retrieval time, source URLs, and snippets are saved and inspectable. Status and sources update while the response streams. Snippets are labeled as untrusted reference material and added to the final user content; the folder's system prompt is unchanged. Only HTTP(S) source links are clickable. No results is shown explicitly; a search error stops generation instead of silently answering without the requested search.
- Retrying after search completed reuses the saved results and exact model input. If search failed or was interrupted before results were saved, retry runs the same saved query again. Later Chat turns retain earlier attachments and search context in history; disabling search prevents a new search, not inclusion of prior context.

| Per-submission limit | Value |
|---|---|
| Files | 4 |
| Original file size | 10 MiB each, 20 MiB combined |
| Image dimensions | 20 million pixels per image |
| PDF pages | 20 combined across all PDFs |
| Extracted PDF text | 100,000 characters combined |
| Rendered PDF image bytes | 32 MiB combined |

Invalid, unreadable, password-required, or over-limit attachments reject the entire submission before any search or model request. Pages/text are never silently truncated; split longer documents yourself. The selected model's context/image limits may be lower and can still cause an upstream rejection.

**Privacy and storage:** only the query is sent to search; the conversation, original images, PDF page renditions, extracted text, and search snippets are sent to the configured model service. SQLite stores these inputs and immutable request history, so image-heavy chats can grow substantially. Original PDF files are not retained; keep them separately. Local storage and the database are not encrypted by this application.

### Folder modes

**Chat** sends the system prompt plus conversation history. A second submission is rejected while generation is in progress, including requests from another tab.

**Memo** saves notes without model calls, including when opening the chat. Converting a memo folder preserves its previous system prompt for later use.

**Translation** splits the input at line or sentence boundaries, with a hard character limit as a fallback. Splitting preserves the source text. Each chunk is translated independently with the system prompt and that chunk only. Source and translated text appear together, in order. The batch continues if the browser closes; failures pause the remaining chunks until retry.

## Reliable generation and prompt history

The send transaction saves user messages and a generation snapshot: model, supported generation parameters, system prompt, attachments, exact ordered API messages, and optional search intent. For search-enabled turns, the worker saves results and final enriched messages atomically **before** contacting the model; until then Prompt labels the request as prepared rather than finalized. Once prepared, model input is immutable. Editing settings or folder types afterward cannot change the stored request.

A background worker generates each response once. Generation state, partial output, reasoning, and errors are stored in SQLite. SSE connections display this state; they never initiate duplicate model calls. There is at most one live SSE connection per conversation view, including translation batches. Escaped HTML and multiline SSE framing preserve line breaks, indentation, and literal HTML in model output.

On reconnection, output is replaced with the stored content so it cannot be appended twice. Completed output is persisted as an assistant message together with token usage. Restarted work is marked interrupted and can be explicitly retried. A retry keeps the original request and increments an attempt counter so a late result from the old worker cannot overwrite it.

Older messages remain available. Their Prompt pages are labeled as best-effort reconstructions when complete request snapshots were not stored. Existing tables and messages are preserved; generation storage, attachment/API-content columns, search-state columns, and indexes are added automatically at startup.

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

Tests use a temporary database and mocked Ollama responses; they do not send prompts to a live service or modify your chat database. They cover generation concurrency, request snapshots, SSE escaping and whitespace, translation pairing, failure/retry/stop behavior, settings validation, deletion, memo isolation, direct links, upstream stream parsing, original image fidelity, PDF page rendering/text extraction, atomic attachment rejection, search opt-in/failure/reuse, and live search-state transitions.

### Browser checks

Playwright is an optional development dependency; the application still needs no frontend build.

```bash
npm install
npx playwright install chromium
npm run test:browser
```

The browser suite starts and stops its own preview server with a temporary database and simulated model responses. It exercises real buttons, typing, dialogs, streaming, drafts, settings, crowded folder menus, and mobile layouts at 320px and 390px. Screenshots go into a unique directory under the ignored `test-results/` directory. On macOS it can use an existing `/Applications/Chromium.app`; `PLAYWRIGHT_CHROMIUM_EXECUTABLE` overrides that choice. Use `HEADED=1 npm run test:browser` to watch the checks.

The interactive coverage inventory and results are in [tests/UI_QA.md](tests/UI_QA.md).

```text
app.py           App setup, startup recovery, full-page navigation
db.py            SQLite storage and atomic generation state transitions
generation.py    Background generation workers, durable search preparation, safe errors
ollama.py        API payloads, web search, response parsing, metadata cache
inputs.py        Validated images, PDF page rendering/text extraction, content parts
templating.py    Shared template helpers and ordered conversation rendering
routers/         Folder, session, chat, stop/retry, and prompt-viewer routes
templates/       Jinja HTML and htmx/SSE fragments
static/app.js    Drafts, keyboard actions, scroll behavior, feedback, sidebar
static/app.css   Responsive workspace styles
tests/           Offline regression suite
```

The principal routes remain HTML/SSE endpoints: `/folders`, `/folders/{id}/sessions`, `/sessions/{id}`, `/sessions/{id}/send`, `/sessions/{id}/stream?since=<user_message_id>`, `/sessions/{id}/memo`, `/sessions/{id}/messages`, and `/sessions/{id}/messages/{id}/prompt`. `/start` creates a chat, `/sessions/{id}/stop` cancels unfinished work, and `/sessions/{id}/retry` retries the latest failed or stopped submission. Only the most recently stored message can be deleted, and deletion is blocked while work is active.
