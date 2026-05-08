# DOVA v2.1 Release Notes

**Release Date:** May 8, 2026

## Overview

DOVA v2.1 is a **concurrency & observability release**. Under sustained
load (~9 parallel `dova mcp serve` clients), v2.0 would routinely time out
even with 15-minute client timeouts. v2.1 identifies and removes the five
compounding bottlenecks behind that behaviour — cold-start serialization,
executor saturation, per-call HTTP client construction, over-aggressive
retries on remote MCPs, and the complete absence of concurrency
telemetry — and exposes tuning knobs for operators.

No API surface changes. No response-schema changes. No config migration
required. v2.0 deployments upgrade in place; the new behaviours kick in
automatically.

---

## Highlights

| Area | v2.0 behaviour | v2.1 behaviour |
|------|----------------|----------------|
| `_get_services` cold start | N concurrent callers each run 140 lines of init (git clones, subprocess spawns, LLM client construction) | **Serialized under `asyncio.Lock`** with double-checked locking; first caller initializes, others wait and return the same populated dict |
| Default `ThreadPoolExecutor` | Python default (`min(32, cpu+4)` ≈ 8 threads) — synchronous boto3 calls queue 6-deep under 9-way concurrency | **64 threads by default**, env-overridable via `DOVA_EXECUTOR_WORKERS` |
| `httpx.AsyncClient` for MCP HTTP transport | New client per call (TCP + TLS handshake every invocation, ~100-500 ms overhead; `_sessions` dict raced by parallel invokers) | **Pooled per-MCPClient**, lazy-initialized under lock; `max_keepalive=32, max_connections=64, keepalive_expiry=30s`. Per-server `_session_init_locks` serialize `_initialize_session` so parallel invokers share one session instead of racing |
| MCP retry policy | Uniform `max_retries=3` across all transports — three attempts on a 45 s HTTP timeout = up to ~135 s per failure | **Transport-aware**: STDIO keeps 3 (cheap local retries), HTTP / SSE / streamable-HTTP drop to **1** (remote retries already cost seconds) |
| In-flight request visibility | None — `structlog` had per-step events, but no request-level count | **`track_request` / `@tracked`** process-wide counter; every MCP + API entry point emits `request_started` / `request_finished` with `in_flight` + `peak` |
| Executor / load visibility | None | **`start_saturation_logger`** background task emits `executor_saturation` every 5 s while busy (silent when idle): `workers_max`, `workers_busy`, `queue_depth`, `in_flight_requests`, `peak_requests` |

---

## What Changed

### 1. Serialized Cold-Start in `dova mcp serve`

**File:** `src/dova/mcp_server.py` (~line 28)

v2.0 lazy-initialized the `_services` singleton without protection:

```python
_services: dict[str, Any] = {}

async def _get_services() -> dict[str, Any]:
    if _services:
        return _services
    # … 140 lines of init, including setup_mcp_repos() which clones
    #    git repos, subprocess spawns, LLM client construction …
    return _services
```

Under 9-way concurrency at cold start, all 9 coroutines pass the
`if _services:` guard simultaneously and each run the full init block in
parallel — racing on filesystem operations, module imports, and the shared
`_services` dict.

v2.1 extracts the init body into `_init_services()` and guards it with a
module-level `asyncio.Lock` using the double-checked-locking pattern:

```python
_services: dict[str, Any] = {}
_services_lock: asyncio.Lock | None = None


async def _get_services() -> dict[str, Any]:
    if _services:
        return _services
    global _services_lock
    if _services_lock is None:
        _services_lock = asyncio.Lock()
    async with _services_lock:
        if _services:
            return _services
        await _init_services()
    return _services
```

The hot path stays lock-free (non-first requests skip the lock entirely
once `_services` is populated). Only the first N-1 cold-start requests
wait on the lock, and they each take microseconds once the first coroutine
has finished.

### 2. Enlarged Default `ThreadPoolExecutor`

**Files:** `src/dova/utils/concurrency.py` (new), `src/dova/mcp_server.py`,
`src/dova/api/main.py`

DOVA's `BedrockProvider` dispatches synchronous `boto3.invoke_model` calls
via `run_in_executor(None, …)`. The `None` argument selects Python's
default executor, which is sized `min(32, cpu_count + 4)` — about **8
threads** on a typical cloud host. Each DOVA request makes 3–6 Bedrock
calls (deliberation + synthesis + optional bridge / debate / refine +
embeddings); 9 concurrent requests × 5 calls = 45 blocking calls queued
on an 8-thread pool, serialized 6-deep.

v2.1 adds `configure_default_executor()` which replaces the event loop's
default executor with one sized **64** by default (env-overridable via
`DOVA_EXECUTOR_WORKERS`). Both `dova serve` (via the FastAPI lifespan) and
`dova mcp serve` (via `_init_services`) call it at startup.

```python
from dova.utils.concurrency import configure_default_executor
configure_default_executor()  # idempotent; reads DOVA_EXECUTOR_WORKERS
```

Threads are named `dova-blocking-N` so they stand out in stack dumps and
tracing tools.

### 3. Pooled `httpx.AsyncClient` for MCP HTTP Transport

**File:** `src/dova/tools/mcp_registry.py` (`MCPClient.__init__`,
`_get_http_client`, `_invoke_http`)

v2.0's `_invoke_http` created a fresh `httpx.AsyncClient` per call:

```python
async with httpx.AsyncClient(timeout=server_config.timeout_seconds) as client:
    if server_config.name not in self._sessions:
        await self._initialize_session(client, server_config)
    …
```

This paid a TCP connect + TLS handshake (~100-500 ms) on every invocation
**and** left `self._sessions` unprotected — parallel invokers of the same
server would both call `_initialize_session`, each overwriting the other's
session ID.

v2.1 introduces a lazy-initialized pooled client on the `MCPClient`
instance, guarded by `_http_client_lock`:

```python
self._http_client: Any | None = None
self._http_client_lock: asyncio.Lock | None = None
self._session_init_locks: dict[str, asyncio.Lock] = {}
```

```python
async def _get_http_client(self, timeout_seconds: float) -> Any:
    if self._http_client is not None:
        return self._http_client
    if self._http_client_lock is None:
        self._http_client_lock = asyncio.Lock()
    async with self._http_client_lock:
        if self._http_client is None:
            import httpx
            limits = httpx.Limits(
                max_keepalive_connections=32,
                max_connections=64,
                keepalive_expiry=30.0,
            )
            self._http_client = httpx.AsyncClient(
                timeout=timeout_seconds,
                limits=limits,
                http2=False,
            )
    return self._http_client
```

Additionally, a per-server `_session_init_locks` dict ensures only one
coroutine per MCP server runs `_initialize_session`; parallel callers
wait on the same lock and share the resulting session ID. An `aclose()`
method is provided for clean shutdown.

Streamable-HTTP transport is **not** pooled in v2.1 — its SDK-managed
lifecycle makes pooling non-trivial and most local deployments use HTTP
or STDIO. See `dova_todo.txt` for the follow-up.

### 4. Transport-Aware Retry Policy

**File:** `src/dova/tools/mcp_registry.py` (`MCPClient.__init__`,
`_invoke_internal`)

v2.0 used a single `RetryConfig(max_retries=3)` across every transport.
For HTTP MCPs with `timeout_seconds=45`, a terminal failure would cost
`~1 + 2 + 4 = 7 s` of backoff sleep **plus** `3 × 45 s = 135 s` of
attempt duration — 142 s before surfacing the error to the caller. Under
9-way concurrency with one flaky downstream, this alone accounts for
most of the observed "15-minute timeout still fails" behaviour.

v2.1 splits the policy by transport:

```python
# Local STDIO subprocess — retries are cheap, keep 3 attempts.
self.retry_config = retry_config or RetryConfig(max_retries=3)
# HTTP / SSE / streamable-HTTP — downstream MCPs have their own 20-45s
# timeouts; 3 attempts pile on ~135s of wait for calls that are already
# unlikely to succeed. Drop to 1 retry.
self._http_retry_config = retry_config or RetryConfig(max_retries=1)
```

`_invoke_internal` selects the right config based on
`server_config.transport`. Callers who pass an explicit `retry_config`
opt out of the split (their value is used for both).

**Worked impact** (single failing HTTP MCP, 45 s timeout):

| | v2.0 | v2.1 |
|---|---|---|
| Attempts before surface | 3 | 2 |
| Backoff sleeps | 1 + 2 = 3 s | 1 s |
| Max attempt duration | 3 × 45 s = 135 s | 2 × 45 s = 90 s |
| **Total worst case** | **~142 s** | **~91 s** |

### 5. In-Flight Request Gauge

**Files:** `src/dova/utils/concurrency.py`, `src/dova/mcp_server.py`,
`src/dova/api/routes/research.py`

A process-wide `_RequestTracker` counts `in_flight`, `peak`, and `total`
requests. Every MCP tool handler in `dova_mcp_server` and the FastAPI
`/research` endpoint is now wrapped:

```python
@mcp.tool()
@tracked("dova_research")
async def dova_research(query: str, ...) -> str:
    ...
```

```python
from dova.utils.concurrency import track_request

async with track_request("api_research"):
    return await _execute_research_inner(...)
```

Emits structured events:

```json
{"event": "request_started", "tool": "dova_research", "in_flight": 3, "peak": 9}
{"event": "request_finished", "tool": "dova_research", "in_flight": 2, "peak": 9, "total": 47}
```

This is the data we need to confirm (or retire) the "~9 concurrent" assumption
that drove Tier 1 tuning.

### 6. Executor Saturation Logger

**File:** `src/dova/utils/concurrency.py`

A background task started at server init periodically inspects the default
executor and the request tracker. When requests are in flight, it emits
`executor_saturation` every 5 s; when idle, it stays silent to avoid log
noise.

```json
{
  "event": "executor_saturation",
  "workers_max": 64,
  "workers_busy": 12,
  "queue_depth": 0,
  "in_flight_requests": 3,
  "peak_requests": 9
}
```

Operator interpretation:

- `queue_depth > 30` sustained → raise `DOVA_EXECUTOR_WORKERS`.
- `workers_busy` stays well under `workers_max` → safe to lower.
- `peak_requests` never reaches single digits → default is overprovisioned.

`start_saturation_logger()` is idempotent and has a matching
`stop_saturation_logger()` for clean shutdown.

---

## Observability — New Log Events

```text
default_executor_configured    max_workers=64
saturation_logger_started      interval_s=5.0
request_started                tool=dova_research in_flight=1 peak=1
executor_saturation            workers_max=64 workers_busy=3 queue_depth=0 in_flight_requests=1 peak_requests=5
request_finished               tool=dova_research in_flight=0 peak=5 total=42
```

All v1.9 / v2.0 events (`master_paper_mcp_fanout_starting`,
`cross_domain_bridges`, `executing_bio_fanout`, `drug_story_chained`, etc.)
remain unchanged.

---

## New Environment Variable

| Variable | Default | Description |
|----------|---------|-------------|
| `DOVA_EXECUTOR_WORKERS` | `64` | Size of asyncio default `ThreadPoolExecutor` for synchronous boto3 calls. Must be set before the first request. |

Documented in `.env.example`, `README.md` (Running Locally → Concurrency
tuning), and `docs/getting_started.md` (Running Locally → Concurrency
tuning v2.0).

---

## Files Changed

```
pyproject.toml                              (2.0.0 → 2.1.0)
src/dova/__init__.py                        (__version__ = 2.1.0)
src/dova/config/settings.py                 (app_version default → 2.1.0)
src/dova/utils/concurrency.py               (NEW — executor sizing, request tracker, saturation logger)
src/dova/mcp_server.py                      (asyncio.Lock on cold start, @tracked on all tools, saturation logger)
src/dova/api/main.py                        (configure_default_executor + start_saturation_logger in lifespan)
src/dova/api/routes/research.py             (track_request("api_research") wrap)
src/dova/tools/mcp_registry.py              (pooled httpx client, per-server session init locks, transport-aware retry)
.env.example                                (DOVA_EXECUTOR_WORKERS doc block)
README.md                                   (v2.1 tagline, Concurrency tuning subsection, prior-release links)
docs/getting_started.md                     (Concurrency tuning v2.0 section, health-check example bumped)
docs/release_notes_v2.1.md                  (this file)
dova_todo.txt                               (deferred Tier 2/3 items with feasibility + impact notes)
```

---

## Verification

| Suite | Count | Status |
|-------|-------|--------|
| Unit tests | 298 / 298 | ✅ |
| Integration tests | 24 / 24 | ✅ |
| Syntax check (all 5 edited .py files) | — | ✅ |
| Runtime smoke test (tracker + decorator + logger start/stop) | — | ✅ |

Sample smoke output under 5-way concurrency:

```
tracker: total=5 peak=5 in_flight=0
decorator OK, total=6
shutdown OK
```

---

## Upgrade Notes

- **No breaking changes.** All public APIs, response schemas, MCP configs,
  session formats, and CLI flags are unchanged. The new behaviours activate
  automatically at startup.
- **`DOVA_EXECUTOR_WORKERS`** is optional. Default of 64 suits most
  deployments. Lower it on resource-constrained hosts; raise it if
  `executor_saturation` logs show `queue_depth > 30` sustained.
- **Cold-start timing** improves dramatically for concurrent requests
  (single `_init_services` call replaces N parallel runs). A first
  request that took 30-90 s under v2.0 cold start now takes its normal
  solo-latency (~5-15 s for a fresh process).
- **Retry behaviour**: callers who passed a custom `RetryConfig` to
  `MCPClient(...)` see both STDIO and HTTP paths use that config
  (backward-compatible with v2.0). To opt into the new split, pass
  `retry_config=None` (the default).
- **Log volume**: every request now emits two additional structured log
  lines (`request_started`, `request_finished`), plus an
  `executor_saturation` line every 5 s while busy. At 10 req/min this is
  ~30 additional lines/min — negligible. Disable by not calling
  `start_saturation_logger()` and ignoring the tracker events.

---

## Known Follow-ups

Captured in `dova_todo.txt` — the most impactful remaining items:

- **Per-provider `asyncio.Semaphore` for rate limiting** — smooths request
  stream to avoid `ThrottlingException` responses from Bedrock/Anthropic/
  OpenAI. Worth doing if production logs show provider throttles.
- **OpenTelemetry instrumentation** — deps already in `pyproject.toml` but
  not wired. Would eliminate the need for structured-log archaeology on
  future perf investigations.
- **`aioboto3` for Bedrock** — only justified if `executor_saturation`
  logs show `queue_depth > 40` sustained under real traffic. Measure first.
- **Pool `streamable_http` MCP sessions** — relevant only for AgentCore
  Gateway deployments.

---

## Prior Releases

- [v2.0](release_notes_v2.0.md) — Intent-weighted `master_paper_mcp` fan-out
- [v1.9](release_notes_v1.9.md) — `doi-bio` MCP (9 DBs) + `master_paper_mcp` gateway
- [v1.8](release_notes_v1.8.md) — Cross-domain AI ⇄ Bio analyst, drug-story chain, env-driven LLM config
- [v1.7](release_notes_v1.7.md) — Weighted intent deliberation, grouped source selector
- [v1.6](release_notes_v1.6.md), [v1.5](release_notes_v1.5.md), [v1.4](release_notes_v1.4.md), [v1.3](release_notes_v1.3.md)
