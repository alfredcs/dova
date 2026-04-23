# DOVA v1.6 Release Notes

**Release Date:** April 23, 2026

## Overview

DOVA v1.6 makes the web UI feel alive. Responses stream in token-by-token, a new **live Thinking sidebar** shows the orchestrator's reasoning as it happens, and answers are now publication-grade: **LaTeX formulas** and **IEEE-style algorithmic pseudocode** are rendered natively instead of being dumped as raw text. Under the hood, the chat pipeline parallelizes context loading and exposes a proper Server-Sent Events endpoint, cutting perceived latency from 20-30 s to first-token-in-seconds.

---

## Highlights

| Area | v1.5 behavior | v1.6 behavior |
|------|--------------|---------------|
| Chat response delivery | One blocking JSON at the end | SSE stream, token-by-token |
| Reasoning visibility | Post-hoc `thinking` list in final payload | Live right-hand sidebar, updates during run |
| Math rendering | Optional, prompt was vague | Mandatory LaTeX (`$…$` / `$$…$$`), rendered with KaTeX |
| Algorithm rendering | "PDL pseudo code" as plain text | IEEE-style numbered algorithm blocks (`\Require`, `\State`, `\For`, `\If`, `\Return`) |
| Context loading | Sequential (user model then conversation) | Parallel (`asyncio.gather`) |

---

## New Features

### 1. Live streaming chat (Server-Sent Events)

New endpoint: **`POST /api/v1/chat/stream`** — mirrors `/api/v1/chat` but returns `text/event-stream`.

Emitted event types:

| Event | Payload | When |
|-------|---------|------|
| `thinking` | `{ step_type, content }` | Deliberation, plans, reflections |
| `stage` | `{ stage, message, ... }` | Pipeline milestones (`deliberating`, `searching`, `synthesizing`) |
| `log` | `{ step, status, elapsed_ms }` | Per-tool timings |
| `tool_complete` | `{ tool, count, items }` | Each arxiv/github/huggingface/web result batch |
| `synthesis_token` | `{ token }` | One LLM token of the final answer |
| `complete` | Full `ChatResponse` payload | End of turn |
| `error` | `{ message }` | Failure |

Example:

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain the Adam optimizer.", "sources": ["arxiv"], "orchestrator": "thinking"}'
```

### 2. Live Thinking sidebar in the web UI

A new right-side `<aside class="thinking-sidebar">` tails the SSE feed in real time. It shows observations, plans, tool calls with timings, and reflections as color-coded entries — so users can see *what* DOVA is doing while it's doing it.

- Toggled by the 💭 header button (persisted in `localStorage`).
- Auto-opens when "Show thinking process" is enabled in Settings.
- Collapses to zero width on wide screens, becomes a slide-over panel below 1024 px.

### 3. Mandatory LaTeX formulas

The synthesis prompt now requires the LLM to emit:

- Inline math: `$O(n \log n)$`, `$\mathcal{L}(\theta) = -\mathbb{E}[\log p_\theta(x)]$`
- Display math (own paragraph): `$$ \mathrm{Attention}(Q,K,V) = \mathrm{softmax}(QK^\top / \sqrt{d_k})\,V $$`

Rendered on the client with KaTeX. Loss functions, scaling laws, gradient updates, complexity bounds, probabilities, and anything with sub/superscripts now appear as proper math, not `O(n log n)` plain text.

### 4. IEEE-style algorithm blocks

The LLM now emits any procedure, training loop, optimization step, decoding strategy, or search method as a fenced block:

````
```algorithm
\Caption{Algorithm 1: Mini-batch SGD with Momentum}
\Require dataset $\mathcal{D}$, learning rate $\eta$, momentum $\beta$, batch size $B$
\Ensure trained parameters $\theta$
\State Initialize $\theta \leftarrow \theta_0$, $v \leftarrow 0$
\For{$t = 1$ to $T$}
    \State Sample batch $\mathcal{B}_t \subset \mathcal{D}$ with $|\mathcal{B}_t| = B$
    \State $g_t \gets \tfrac{1}{B} \sum_{x \in \mathcal{B}_t} \nabla_\theta \mathcal{L}(\theta; x)$
    \State $v \gets \beta\, v + (1 - \beta)\, g_t$
    \State $\theta \gets \theta - \eta\, v$
\EndFor
\Return $\theta$
```
````

The UI renders this as an IEEE `algorithmic` environment with:

- Auto-incremented line numbers (except `\Require` / `\Ensure` / `\Input` / `\Output`).
- Indentation tracked across `\For`/`\EndFor`, `\If`/`\Else`/`\EndIf`, `\While`/`\EndWhile`, `\Function`/`\EndFunction`.
- Keyword styling (pink monospace) + inline KaTeX for `$…$` math on every line.
- Caption line as a bold header.

No more Python stand-ins for methodology — core algorithms are presented as they would be in a paper.

### 5. Parallel context loading

`ThinkingOrchestrator.execute()` now loads the user model and conversation context concurrently with `asyncio.gather`, shaving one round-trip off every request.

---

## Architecture

### New streaming data flow

```
Client (index.html)
    │ POST /api/v1/chat/stream  (fetch + ReadableStream)
    ▼
chat.stream_message
    │ creates InteractiveSession (reuses orchestrator)
    │ spawns _runner() task with progress_cb
    ▼
ThinkingOrchestrator.execute(task, progress=progress_cb)
    │ emits progress: thinking | stage | log | tool_complete | synthesis_token
    ▼
asyncio.Queue → event_generator() → StreamingResponse
    │ SSE frames: "event: <type>\ndata: <json>\n\n"
    ▼
Client parses frames, updates sidebar + streams tokens into assistant bubble.
```

### Frontend rendering pipeline

```
raw markdown from LLM
    │
    ├─ extract ```algorithm blocks           → renderAlgorithmBlock()
    ├─ extract $$…$$ / $…$ math              → KaTeX.renderToString()
    ├─ marked.parse(remaining markdown)       → HTML
    └─ reinsert rendered algorithms + math    → final HTML
```

---

## Files Changed

### Modified

| File | Changes |
|------|---------|
| `pyproject.toml` | Version bump: `1.3.0` → `1.6.0` |
| `src/dova/__init__.py` | `__version__` → `"1.6.0"` |
| `src/dova/config/settings.py` | `app_version` default → `"1.6.0"` |
| `src/dova/agents/thinking_orchestrator.py` | Parallel context loading; rich `thinking` / `action` / `reflection` progress events; strengthened synthesis prompt (mandatory LaTeX + IEEE-algorithm rules with worked example) |
| `src/dova/api/routes/chat.py` | New `POST /api/v1/chat/stream` SSE endpoint that reuses the existing session store and orchestrator progress callback |
| `src/dova/api/static/index.html` | Live Thinking sidebar (markup + CSS); streaming client with `fetch` + `ReadableStream`; algorithm block renderer (`renderAlgorithmBlock`); streaming caret; thinking step type styles (`reflection`, `deliberation`, `tool`, `stage`) |

### New

| File | Description |
|------|-------------|
| `docs/release_notes_v1.6.md` | This document |

---

## Usage

### Web UI

1. Open the DOVA UI (served at `/` by the API).
2. Click 💭 in the header to open the Thinking sidebar (persists across reloads).
3. Ask a research question — watch the sidebar populate with observations, plans, and tool timings while the answer streams into the chat bubble.
4. Any equation or algorithm in the answer renders automatically (KaTeX + IEEE algorithm block).

### API (streaming)

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain speculative decoding and give the algorithm.",
    "sources": ["arxiv"],
    "orchestrator": "thinking",
    "show_thinking": true
  }'
```

Stream frames arrive as `event: <type>\ndata: <json>\n\n`.

### API (non-streaming, unchanged)

The legacy `POST /api/v1/chat` and `POST /api/v1/chat/upload` endpoints are untouched — existing integrations continue to work.

---

## Compatibility

- **Backwards compatible.** No breaking changes to `/api/v1/chat`, `/api/v1/chat/upload`, or session schemas.
- File uploads continue to use the non-streaming path (multipart + SSE combo intentionally avoided).
- `show_thinking=false` still works — the sidebar simply stays empty.

---

## Validation

- **Unit tests:** 225 passing (`pytest tests/unit`).
- **SSE smoke test:** `/api/v1/chat/stream` verified end-to-end with a mocked orchestrator:
  - Correct `text/event-stream` content-type.
  - All event types flow through (`thinking`, `stage`, `synthesis_token`, `complete`).
  - Tokens concatenate into the final answer.
  - Algorithm fence and inline LaTeX preserved in the `complete` payload.
  - Thinking steps captured into `ConversationTurn.thought_chain`.
- **HTML:** tag balance verified on `index.html`.

---

## Known Limitations

- The upload endpoint (`/api/v1/chat/upload`) does not stream — attaching files reverts to a buffered JSON response.
- Algorithm renderer supports the canonical LaTeX `algorithmic` keywords; exotic macros (`\Procedure`, `\Call`) render as plain text if the LLM uses non-standard syntax.
- Sidebar thinking feed is not persisted per session — it resets on each query (scrollable history stays in the final message payload).

---

## What's Next (v1.7 roadmap)

- Streaming file uploads (multipart → SSE bridge).
- Persisted thinking history per session with a "replay reasoning" view.
- Inline citations on LaTeX-rendered equations (hover for source).
- Algorithm export as standalone `.tex` snippets.
- Sidebar search and filtering across past reasoning steps.

---

## Contributors

- DOVA Team

---

## Changelog Summary

| Category | Changes |
|----------|---------|
| Features | SSE streaming chat, live Thinking sidebar, IEEE algorithm renderer, mandatory LaTeX formulas |
| Performance | Parallel user-model + context loading; first-token latency in seconds vs. 20-30 s full-response wait |
| API | New `POST /api/v1/chat/stream`; richer `progress` events (`thinking`, `action`, `reflection`) from `ThinkingOrchestrator` |
| Frontend | Right-side Thinking sidebar; `fetch + ReadableStream` SSE client; `renderAlgorithmBlock()`; streaming caret |
| Prompting | Synthesis prompt rewritten: mandatory LaTeX math delimiters, mandatory IEEE-style `algorithm` fenced blocks with worked example |
| Versioning | `pyproject.toml`, `dova.__version__`, `Settings.app_version` all bumped to `1.6.0` |
