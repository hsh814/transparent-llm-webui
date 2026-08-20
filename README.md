# transparent-llm-webui

Minimal single-user internal AI chat app. FastAPI + htmx 2.0 (SSE streaming) + SQLite, chatting with Ollama Cloud (`https://ollama.com`). No login, no build step, no JS framework.

## Features

- Folders with an optional **system prompt** — always visible in the UI (open `<details>` block), never injected invisibly. The exact text shown is what the model receives.
- Per-chat model choice + generation params (reasoning effort, temperature, top_p, max_tokens, num_ctx, top_k, repeat_penalty, seed), persisted per session and shown as a badge on every assistant message.
- Token streaming over SSE: thinking text fills a "Thinking" region, then the answer streams in.
- Token usage tracking: per-message token badges (prompt/completion/total) and a running session total in the chat header.
- Copy buttons: per message and **Copy all** in the header.
- A built-in **Memo** folder for quick notes — saved per chat, shown like messages, never sent to a model.
- Folders can be **pinned** to the top of the sidebar.
- SQLite storage (stdlib `sqlite3`, no ORM).

## Setup

```bash
cd transparent-llm-webui
uv sync
```

Create `.env` in the repo root:

```
OLLAMA_API_KEY=your-key
```

The key is required — the app fails fast at startup if it's missing.

## Run

```bash
uv run uvicorn app:app --port 8000
```

Open `http://localhost:8000`. Single-user, no auth — do not expose on a public interface.

## Usage

1. **+ Folder** — create a folder; the **⋮** menu converts it to **Chat / Memo / Translation** or deletes it; open **Edit** to name it, set its system prompt, and (for translation folders) the chunk char limit.
2. **+ New chat** — creates a session in the folder.
3. Pick the model in the header dropdown; adjust params and hit **Apply**.
4. Type a message and **Send**. The assistant's thinking and answer stream in; the model + effort badge is stamped on the bubble.
5. In a **Memo** folder, the composer **Save** button stores a note — memo messages are displayed but never sent to a model.
6. In a **Translation** folder, the composer **Translate** button splits the input by line/punctuation into chunks of at most the folder's chunk char limit. Each chunk is translated independently (system prompt + chunk only, no conversation history) and the results appear sequentially as separate bubbles.

Changing the model mid-chat only affects subsequent messages — past badges stay accurate.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `OLLAMA_API_KEY` | — (required) | Bearer token for Ollama Cloud |
| `OLLAMA_API_BASE_URL` | `https://ollama.com` | Native API (`/api/show`) |
| `OLLAMA_OPENAI_BASE_URL` | `https://ollama.com/v1` | OpenAI-compatible API (`/v1/models`, `/v1/chat/completions`) |

## Architecture

```
app.py            FastAPI app: env load, static mount, router includes, GET /
db.py             SQLite schema + access functions (single connection + lock)
ollama.py         Typed client over Ollama Cloud (list_models, chat_stream, chat_once, cached)
templating.py     Jinja2 env + shared fragment helpers and globals
routers/
  folders.py      Folder CRUD, pin, system-prompt partial
  sessions.py     Session create/rename/delete, model+params update, model refresh, chat surface
  chat.py         Send (POST), SSE stream (GET), memo (POST), message delete, history refresh
templates/        Jinja2: base, index, chat surface, partials
static/           htmx.min.js, sse.js (vendored), app.css
chat.db           SQLite database (created on first boot, gitignored)
```

### How streaming works

The htmx 2.0 SSE extension only creates native `EventSource` (GET) connections — it does not consume SSE returned from an `hx-post`. So:

1. `POST /sessions/{id}/send` persists the user message and returns the user bubble + an empty assistant bubble whose root carries `sse-connect="/sessions/{id}/stream?since=<user_msg_id>"`.
2. htmx swaps the bubbles in; the extension opens the EventSource.
3. `GET /sessions/{id}/stream` runs the generation and emits `reasoning` / `content` events that append into the bubble's zones, persists the assistant message, then emits `done` — which closes the source (`sse-close`) and swaps in the finalized bubble.

Reconnects are safe: if an assistant message already exists for the `since` id, the stream replays it from the DB instead of re-running the model.

### System prompt transparency

The folder row is the source of truth. On each send, the prompt is mirrored into exactly one `role=system` message row (updated in place, always the first row) and sent verbatim as the first API message. The same text is rendered in the UI system-prompt block and the system message bubble. No hidden or additional prompt is ever sent. Prompt texts are content-addressed in `system_prompts` (sha256); each user/assistant message records the hash of the prompt in effect at send time (`messages.system_prompt_hash`), so changing the folder prompt only affects new turns and every assistant bubble's Prompt page reconstructs the exact prompt it received.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Full page |
| POST | `/folders` | Create folder (form: `name`, `system_prompt`) |
| POST | `/folders/{id}/update` | Update folder (form: `name`, `system_prompt`, `chunk_limit`, `active_session_id`) |
| POST | `/folders/{id}/delete` | Delete folder + cascade |
| POST | `/folders/{id}/pin` | Toggle folder pin (sidebar order) |
| POST | `/folders/{id}/convert` | Convert folder type (form: `folder_type` ∈ chat/memo/translation) |
| GET | `/folders/{id}/system-prompt` | System-prompt partial |
| POST | `/folders/{folder_id}/sessions` | Create session |
| POST | `/sessions/{id}/rename` | Rename (form: `title`) |
| POST | `/sessions/{id}/delete` | Delete session + cascade |
| POST | `/sessions/{id}/model` | Update model + params (form fields) |
| POST | `/sessions/{id}/model/refresh` | Clear model cache, re-fetch models + ctx cap |
| GET | `/sessions/{id}` | Chat surface fragment |
| POST | `/sessions/{id}/send` | Send message → bubbles HTML (chat: + SSE stream bubble; translation: chunked user bubbles + pending translate bubble) |
| POST | `/sessions/{id}/translate` | Translate one chunk (form: `user_message_id`); returns finalized bubble + next pending bubble |
| POST | `/sessions/{id}/memo` | Save memo (memo folders only, 403 otherwise) |
| GET | `/sessions/{id}/stream?since=` | SSE stream (EventSource) |
| POST | `/sessions/{id}/messages/{mid}/delete` | Delete the session's last message (403 otherwise) |
| GET | `/sessions/{id}/messages/{mid}/prompt` | Full-page reconstruction of the prompt sent for that assistant turn |
| GET | `/sessions/{id}/messages` | Message history partial |

## Notes

- New sessions inherit the model and params of the most recently edited session in the folder; the schema default is `gemma4:31b`. The model dropdown is populated live from `GET /v1/models` (cached in memory, refreshable via the model selector's Refresh button), so the app degrades gracefully if a model disappears.
- SQLite is accessed under a single module-level connection guarded by a lock — adequate for single-user use.
