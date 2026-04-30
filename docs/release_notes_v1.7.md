# DOVA v1.7 Release Notes

**Release Date:** April 30, 2026

## Overview

DOVA v1.7 makes the orchestrator **understand query intent as a distribution** rather than a single label. Every query is scored across the three top-level groups — **AI**, **Bio**, and **Web** — into a percentage distribution (for example `60% AI / 30% Bio / 10% Web`). Those weights flow through deliberation, synthesis aggregation, and every surfacing layer (CLI, SDK, web UI, SSE events). In the same release we finish the biotech/pharma integration started in v1.6: three hosted MCP endpoints (**PubMed**, **ClinicalTrials.gov**, **PubChem**) are registered out of the box, routed via a single `bio` umbrella with semantic multi-server fan-out, and surfaced in the redesigned grouped source selector.

A substantial plumbing effort also unifies the three user-facing entry points: `dova interact`, `dova research`, and the `dova serve` chat endpoints now share a single `ThinkingOrchestrator` path, so behaviour and observability are identical regardless of surface.

---

## Highlights

| Area | v1.6 behaviour | v1.7 behaviour |
|------|----------------|----------------|
| Query intent | Single winning category (news / technical / ML / etc.) | Percentage distribution over `{ai, bio, web}`, sums to 1.0 |
| Result aggregation in synthesis | Hard `[:5]` slice per source | **Weighted**: slots per group scale with intent, minimum 2 items per active group |
| Web inclusion | Binary (picked or skipped) | **Floor of 10 %** whenever user allows web — general context always present |
| Domain coverage | ArXiv + GitHub + HuggingFace + Web | + **PubMed**, **ClinicalTrials.gov**, **PubChem** under a single `bio` umbrella |
| Source selector UI | 4 flat checkboxes | **3 grouped chips** (AI / Web / Bio) that expand to concrete sources |
| Entry-point parity | `dova research` used task-graph; CLI/UI differed | **All three surfaces** (`interact`, `research`, `serve`) default to `ThinkingOrchestrator` |
| Bio server selection | N/A | Keyword-scored **multi-server fan-out**: cross-domain queries hit multiple bio backends in parallel |

---

## New Features

### 1. Semantic intent weighting

`compute_intent_weights(query, allowed_groups)` in `dova.agents.thinking_orchestrator` produces a distribution like:

```python
>>> compute_intent_weights(
...     "comparison of LLMs and protein transformer models for drug discovery"
... )
{"ai": 0.60, "bio": 0.30, "web": 0.10}

>>> compute_intent_weights(
...     "RFdiffusion de novo binders AF2-Multimer affinity epitopes"
... )
{"ai": 0.05, "bio": 0.85, "web": 0.10}

>>> compute_intent_weights(
...     "what news was announced about OpenAI today"
... )
{"ai": 0.05, "bio": 0.05, "web": 0.90}
```

Properties:

- Keyword-scored across three compact vocabularies (AI / Bio / Web) — no LLM round-trip, runs on every query.
- Floors applied by **deficit redistribution** from over-funded groups, so the distribution still sums to exactly 1.0 — no post-hoc renormalisation that would erase the floors.
- **Web floor 10 %** when web is allowed (user requirement: every answer gets broader context).
- **Group floor 5 %** for any other allowed group with zero keyword hits — prevents a legitimate user selection from being zero-summed out.
- Single-group allowed set (e.g. the user ticked only Bio) yields `{group: 1.0}` with no floors injected.

### 2. Weighted synthesis aggregation

The synthesis prompt builder (`_build_synthesis_prompt`) now divides a fixed 15-slot budget proportionally across the active groups:

| Weight | Slots (AI channels split across papers/repos/models) |
|--------|------------------------------------------------------|
| 60 % AI | ~9 AI items (split ≈ 3 papers + 3 repos + 3 models) |
| 30 % Bio | char-budget for bio MCP output scales to ~750 chars per server |
| 10 % Web | 2 web snippets |

Guarantees:

- Every active group gets a **minimum of 2 items** when it has results — no group can be starved by low weight.
- The synthesis prompt explicitly tells the LLM the distribution so narrative emphasis tracks the weights.
- Bio MCP result char-cap scales with the bio weight (`max(400, 2500 × bio_weight)` chars per server).

### 3. Biotech / pharma MCP integration

Three hosted Streamable-HTTP endpoints are registered by default in the MCP registry:

| DOVA server | URL | Primary tool | Routing triggers |
|-------------|-----|--------------|------------------|
| `pubmed-bio` | `https://pubmed.caseyjhand.com/mcp` | `pubmed_search_articles` | PubMed, PMID, MeSH, biomedical literature, genes/proteins/diseases (default) |
| `clinicaltrials-bio` | `https://clinicaltrials.caseyjhand.com/mcp` | `clinicaltrials_search_studies` | clinical trial, NCT, phase I–IV, enrollment |
| `pubchem-bio` | `https://pubchem.caseyjhand.com/mcp` | `pubchem_search_compounds` | SMILES, InChI, CID, compound, ADMET |

All three are Apache-2.0, reachable without authentication, and validated at registration time.

The orchestrator surfaces a single `bio` umbrella to the deliberation LLM — the 3 sub-servers never bloat the deliberation prompt. At execution time, `_select_bio_servers` performs **keyword-scored semantic fan-out**: single-domain queries route to one server, cross-domain queries (e.g. "PubMed review of BRAF clinical trials") fan out to multiple bio endpoints in parallel.

### 4. Grouped source selector

The web UI's source chips collapse from four flat checkboxes (`arxiv / github / huggingface / web`) into three grouped chips:

- **AI** → `arxiv, github, huggingface`
- **Web** → `web`
- **Bio** → `bio` (routed internally)

A small `expandSourceGroups` helper maps chips to concrete source IDs at the request boundary. Sidebar and dashboard (React app) use the same model, and the live static UI at `src/dova/api/static/index.html` mirrors it.

### 5. Unified orchestrator across all entry points

`dova interact`, `dova research`, and the API's `/api/v1/chat*` + `/api/v1/research*` routes all default to `ThinkingOrchestrator`. Every surface now sees the same deliberation output, the same intent weights, and the same bio fan-out behaviour.

---

## Observability

Every layer exposes intent weights:

- **SSE `stage` event** — `stage: "deliberating"` now includes `intent_weights`.
- **SSE `thinking` event** — a `deliberation`-type step with `"Semantic intent: 60% AI, 30% BIO, 10% WEB"` fires right after deliberation.
- **SSE `complete` payload** — `intent_weights` field on the final envelope.
- **Non-streaming `ChatResponse.intent_weights`** — new Pydantic field on the response schema.
- **SDK** — `ThinkingOrchestrator.extract_research_data(...)` returns `deliberation.intent_weights` to library consumers.
- **`dova interact --verbose`** — prints `Intent: 60% AI, 30% BIO, 10% WEB` in the thought stream.
- **`dova research`** — text output now starts with an `Intent:` line above the synthesized answer.
- **Web UI** — coloured chip row beneath each answer (AI indigo, Bio green, Web blue, `[GROUP N%]`).

Bio fan-out is traceable end-to-end:

```text
executing_bio_fanout           servers=['pubmed-bio', 'clinicaltrials-bio'] tool=bio
bio_server_call_starting       server=pubmed-bio tool=pubmed_search_articles
bio_server_call_complete       server=pubmed-bio success=True data='2340 chars'
bio_server_call_starting       server=clinicaltrials-bio tool=clinicaltrials_search_studies
bio_server_call_complete       server=clinicaltrials-bio success=True data='1820 chars'
```

---

## API changes

### `ChatResponse` (`POST /api/v1/chat` and `/chat/upload`)

New field:

```json
{
  "intent_weights": { "ai": 0.6, "bio": 0.3, "web": 0.1 }
}
```

### SSE `complete` payload (`POST /api/v1/chat/stream`)

`intent_weights` is now included on the envelope — unchanged shape otherwise.

### SSE `stage` payload (`stage: "deliberating"`)

```json
{
  "stage": "deliberating",
  "message": "Decided to search arxiv, github, huggingface, web, bio",
  "action": "use_tools",
  "tools_planned": ["arxiv", "github", "huggingface", "web", "bio"],
  "intent_weights": { "ai": 0.6, "bio": 0.3, "web": 0.1 }
}
```

### `ChatRequest.sources` default

Extended to include `bio`:

```json
{"sources": ["arxiv", "github", "huggingface", "web", "bio"]}
```

Clients that omit `sources` now get the biomedical umbrella automatically.

---

## UI changes

### Static single-page UI (`src/dova/api/static/index.html`)

- Replaced the four flat source chips (`ArXiv / GitHub / HuggingFace / Web`) with three grouped chips (`AI / Web / Bio`, all active by default).
- Added `SOURCE_GROUP_MAP` that expands group IDs into concrete source names at request time.
- Added `buildIntentWeightsCard(weights)` that renders a coloured chip row beneath each assistant message.
- Welcome screen copy updated to mention biomedical sources.
- Cache headers on `GET /` set to `no-store, no-cache, must-revalidate` so UI updates land immediately after a deploy.
- 12 new sample questions across six biomedical categories (Drug Discovery, Clinical Trials, Genomics & Precision Medicine, Protein Design, ML for Medicine, Pharmacology). Total pool: 32 questions.

### React frontend (`frontend/src/`)

- `SearchFilters.tsx`: selection state now stores group IDs (`ai` / `web` / `bio`). Exports `SOURCE_GROUPS` and `expandSourceGroups` used at the API boundary.
- `Dashboard.tsx` / `Sidebar.tsx`: aligned defaults to `['ai', 'web', 'bio']`.
- `api/types.ts` (behaviour unchanged) — the new `intent_weights` field is forward-compatible.

---

## CLI / SDK

- `dova interact --verbose` prints an `Intent: …` line in the thought stream.
- `dova research` text output begins with `Intent: 60% AI, 30% BIO, 10% WEB` above the synthesized answer.
- `dova research --sources` accepts `bio` as a source name.
- `dova_eva` and other SDK consumers see `deliberation.intent_weights` in `extract_research_data(...)` output.

---

## Infrastructure

- **Managed repo robustness**: `MCPRepoManager._update_repo` now runs `git reset --hard HEAD` before `git pull --ff-only`. Fixes a startup failure where `uv pip install -e` had modified the cloned repo's `uv.lock`, leaving it dirty for subsequent pulls.
- **HTML no-store**: root UI response now sets `Cache-Control: no-store, no-cache, must-revalidate` so browsers never serve stale `index.html` after a deploy.
- **Request-level logging**: new `chat_stream_request sources_from_client=…` log at the API entry point makes UI-vs-backend source mismatches trivial to diagnose.

---

## Tests

- **267 unit tests passing**, up from 255 at the start of the v1.7 cycle.
- **37 new tests** covering bio MCP registration, keyword routing, bio server fan-out, `search_bio_tool` dispatch, and intent-weight algorithm (sum-to-1.0 invariant, web floor, group floor, zero-sum prevention, mixed 60/30/10 case).

```bash
$ pytest tests/unit -q
... 267 passed in 1.97s
```

---

## Files changed

```
src/dova/agents/thinking_orchestrator.py     (weighted synthesis + intent scorer + bio fan-out observability)
src/dova/config/mcp_servers.py               (3 bio MCP configs + BIO_MCP_SERVERS list + registry hook)
src/dova/tools/research_tools.py             (search_bio_tool helper + dispatch)
src/dova/services/mcp_repo_manager.py        (git reset --hard before pull)
src/dova/api/main.py                         (no-cache headers on /)
src/dova/api/routes/chat.py                  (intent_weights in ChatResponse + SSE complete + request log)
src/dova/api/schemas/chat.py                 (intent_weights Pydantic field + default includes bio)
src/dova/api/static/index.html               (grouped chips + intent card + biomed samples)
src/dova/cli/main.py                         (Intent: line in `dova research` output)
src/dova/cli/interact.py                     (Intent line + intent_weights on ConversationTurn result)
frontend/src/components/search/SearchFilters.tsx   (group-ID model + expandSourceGroups)
frontend/src/components/layout/Sidebar.tsx         (uses shared SOURCE_GROUPS)
frontend/src/pages/Dashboard.tsx             (default sources = group IDs, expand at API boundary)
tests/unit/agents/test_bio_routing.py        (37 new tests)
docs/bio-mcp-servers-dir.md                  ("Integrated into DOVA" section)
docs/release_notes_v1.7.md                   (this file)
DOVA.md                                      (weighted intent deliberation section)
README.md                                    (v1.7 tagline + feature highlights)
```

---

## Upgrade notes

- **No breaking API changes.** Existing clients continue to work. The added `intent_weights` field on `ChatResponse` and the `complete` SSE payload is additive.
- **Default source list broadened.** Clients that do not send `sources` now get `bio` added to their request. If you want to exclude biomedical results, send an explicit `sources` list without `"bio"`.
- **Browser cache**: after upgrading, hard-refresh (Cmd/Ctrl-Shift-R) once to pick up the new grouped-chip UI. Subsequent deploys are covered by the `no-store` headers.
- **No migration required.** All changes are in-process; no database schemas, MCP configs, or session formats were altered.
