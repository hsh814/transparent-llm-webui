# Transparent Chat

A single-user workspace for conversations, translation, and notes. FastAPI, htmx, and SQLite; no frontend build step or JavaScript framework. Model requests go to Ollama Cloud.

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
- Titles are generated from the first message. Click a title to rename it; changes save on blur or Enter.
- Use the folder menu to choose **Chat**, **Memo**, or **Translation**. Folder settings contain the name and system prompt, plus the translation chunk size.
- The current model stays in the header. Open **Settings** to adjust reasoning effort, temperature, top P, maximum output tokens, or the random seed; click **Apply** to save.
- A configured system prompt is shown above the conversation. It is sent verbatim. Empty prompts add no system message.
- On a desktop, Enter sends a chat message and Shift + Enter inserts a newline. Notes and translations use Enter for newlines and `⌘/Ctrl Enter` to submit. On touch devices, Enter inserts a newline.
- Drafts are stored per chat in browser local storage and cleared only after a successful submission. They survive navigation and reloads, including failed sends.
- **Stop** stops accepting response output and cancels queued translation chunks. Partial output remains visible. **Retry with original settings** retries the latest unfinished submission using its stored request. A stopped upstream connection closes when the worker next receives data or times out.
- Reading earlier messages does not force-scroll to the bottom. **Latest message** returns to the newest output. The composer stays visible on mobile.
- Each assistant message has **Copy** and **Prompt** actions. Prompt opens the saved request and settings, including responses without token usage. **Copy chat** copies the displayed conversation.
- Links to `/?session=<id>` open a complete page and survive reloads and browser navigation.

### Folder modes

**Chat** sends the system prompt plus conversation history. A second submission is rejected while generation is in progress, including requests from another tab.

**Memo** saves notes without model calls, including when opening the chat. Converting a memo folder preserves its previous system prompt for later use.

**Translation** splits the input at line or sentence boundaries, with a hard character limit as a fallback. Splitting preserves the source text. Each chunk is translated independently with the system prompt and that chunk only. Source and translated text appear together, in order. The batch continues if the browser closes; failures pause the remaining chunks until retry.

## Reliable generation and prompt history

The send transaction saves user messages and an immutable generation snapshot: model, supported generation parameters, system prompt, and exact ordered API messages. Editing settings or folder types afterward cannot change a submitted request or its Prompt page.

A background worker generates each response once. Generation state, partial output, reasoning, and errors are stored in SQLite. SSE connections display this state; they never initiate duplicate model calls. There is at most one live SSE connection per conversation view, including translation batches. Escaped HTML and multiline SSE framing preserve line breaks, indentation, and literal HTML in model output.

On reconnection, output is replaced with the stored content so it cannot be appended twice. Completed output is persisted as an assistant message together with token usage. Restarted work is marked interrupted and can be explicitly retried. A retry keeps the original request and increments an attempt counter so a late result from the old worker cannot overwrite it.

Older messages remain available. Their Prompt pages are labeled as best-effort reconstructions because the previous schema did not store complete request snapshots. Existing tables and messages are preserved; the generation table and indexes are added automatically at startup.

### Supported model settings

The app uses Ollama's `/v1/chat/completions` endpoint. Seed is sent as a top-level field. Context size, top K, and repeat penalty are not exposed as controls because this endpoint does not document support for those fields; old stored values are retained but omitted from new requests. See [Ollama's supported request fields](https://docs.ollama.com/api/openai-compatibility).

Context capacity is informational when it can be retrieved. Reasoning-level support depends on the model. The selected model remains available in the selector even if it disappears from the catalog. Metadata requests use short timeouts; successful lookups are cached for five minutes and failures for 30 seconds. The refresh button clears the cache.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_API_KEY` | Required | Ollama Cloud bearer token |
| `OLLAMA_API_BASE_URL` | `https://ollama.com` | Native model metadata API |
| `OLLAMA_OPENAI_BASE_URL` | `https://ollama.com/v1` | Model list and chat completions |
| `CHAT_DB_PATH` | `chat.db` beside `db.py` | Database path; useful for isolated development |

Back up `chat.db` while the application is stopped to preserve your conversations and request history.

## Development and verification

```bash
uv run python -m unittest discover -s tests -v
node --check static/app.js
```

Tests use a temporary database and mocked Ollama responses; they do not send prompts to a live service or modify your chat database. They cover generation concurrency, request snapshots, SSE escaping and whitespace, translation pairing, failure/retry/stop behavior, settings validation, deletion, memo isolation, direct links, and upstream stream parsing.

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
generation.py    Background generation workers and safe error messages
ollama.py        API payloads, response parsing, metadata cache
templating.py    Shared template helpers and ordered conversation rendering
routers/         Folder, session, chat, stop/retry, and prompt-viewer routes
templates/       Jinja HTML and htmx/SSE fragments
static/app.js    Drafts, keyboard actions, scroll behavior, feedback, sidebar
static/app.css   Responsive workspace styles
tests/           Offline regression suite
```

The principal routes remain HTML/SSE endpoints: `/folders`, `/folders/{id}/sessions`, `/sessions/{id}`, `/sessions/{id}/send`, `/sessions/{id}/stream?since=<user_message_id>`, `/sessions/{id}/memo`, `/sessions/{id}/messages`, and `/sessions/{id}/messages/{id}/prompt`. `/start` creates a chat, `/sessions/{id}/stop` cancels unfinished work, and `/sessions/{id}/retry` retries the latest failed or stopped submission. Only the most recently stored message can be deleted, and deletion is blocked while work is active.
