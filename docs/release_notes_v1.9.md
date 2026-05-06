# DOVA v1.9 Release Notes

**Release Date:** May 5, 2026

## Overview

DOVA v1.9 broadens the research surface with two **additive, best-effort** paper-discovery sources:

1. A new **`doi-bio` MCP server** (tfscharff/doi-mcp) that unifies 9 academic databases behind `findVerifiedPapers` and adds a dedicated `verifyCitation` anti-hallucination tool.
2. A new **`master_paper_mcp` gateway integration** in the `ThinkingOrchestrator`: when the gateway is healthy, active umbrellas (`ai` / `bio` / `web`) fan out `search_papers` over subject shards in parallel with the existing tools.

Both sources are strictly additive — when they're unreachable, orchestration degrades silently to the v1.8 behaviour. No API schemas, config files, or session formats change.

---

## Highlights

| Area | v1.8 behaviour | v1.9 behaviour |
|------|----------------|----------------|
| Bio fan-out sources | PubMed + ClinicalTrials + PubChem | **+ `doi-bio` (9 DBs: CrossRef, OpenAlex, PubMed, Semantic Scholar, DBLP, zbMATH, ERIC, HAL, INSPIRE-HEP)** |
| Citation verification | Not exposed | **`verifyCitation` tool** via `search_bio_tool(domain="citation")` — anti-hallucination check |
| Bio semantic router | Three-way (literature / trials / compounds) | **Four-way** — new keywords (`doi`, `crossref`, `openalex`, `verify citation`, …) route to `doi-bio` |
| Paper search gateway | None | **`master_paper_mcp` fan-out** over umbrella subjects in parallel with existing tools, health-cached (30 s) and silently skipped when down |
| `search_bio_tool` domains | `auto / literature / trials / compounds` | **+ `verified_papers` / `doi` / `citation` / `verify`** |
| Bio budget / bridges / top-papers | Recognised `pubmed-bio` / `clinicaltrials-bio` / `pubchem-bio` | **Extended to recognise `doi-bio`** in bridge gating, char-budget allocation, and bio block rendering |

---

## New Features

### 1. `doi-bio` MCP Server — Cross-Database Verified Paper Search

A new `MCPServerConfig` (`BIO_DOI_MCP` in `src/dova/config/mcp_servers.py`) registers **tfscharff/doi-mcp** as a zero-config STDIO server:

```python
BIO_DOI_MCP = MCPServerConfig(
    name="doi-bio",
    transport=MCPTransport.STDIO,
    command="npx -y github:tfscharff/doi-mcp",
    priority=2,
    tools=[... findVerifiedPapers, verifyCitation ...],
)
```

The server joins the canonical bio roster:

```python
BIO_MCP_SERVERS = [
    "pubmed-bio",
    "clinicaltrials-bio",
    "pubchem-bio",
    "doi-bio",       # ← new in v1.9
]
```

**Tools exposed:**

| Tool | Purpose | Params |
|---|---|---|
| `findVerifiedPapers` | Fan-out search across 9 academic DBs (CrossRef, OpenAlex, PubMed, Semantic Scholar, DBLP, zbMATH, ERIC, HAL, INSPIRE-HEP) for papers with verified DOIs | `query`, `source` (default `all`), `limit` (max 20), `yearFrom`, `yearTo` |
| `verifyCitation` | Anti-hallucination check — does this citation exist anywhere across the 9 DBs? | `title`, `authors`, `year`, `doi`, `journal` |

Zero install burden: the registry registers it automatically; the first invocation `npx`-resolves it.

### 2. Bio Semantic Router — DOI / Cross-DB Keywords

`ThinkingOrchestrator._select_bio_servers()` and `_score_bio_server()` both learned a new `doi-bio` keyword set so natural-language queries get routed correctly:

```
doi, crossref, openalex, semantic scholar, dblp, zbmath,
inspirehep, inspire-hep, eric database, hal,
verify citation, verified paper[s], citation check,
citation verification, anti-hallucination
```

Routing is additive: PubMed is still always included in the bio fan-out (v1.8 contract preserved), and queries like *"crossref verified papers for DOI 10.1038/nature12373"* now hit `{doi-bio, pubmed-bio}` in parallel.

`_get_mcp_tool_params("doi-bio", "findVerifiedPapers", query)` returns `{"query": query, "source": "all", "limit": 10}` so the fan-out hits all 9 DBs by default.

### 3. `search_bio_tool` — New Domains

`src/dova/tools/research_tools.py::search_bio_tool` gains four new `domain` values:

| `domain` | Server | Tool |
|---|---|---|
| `verified_papers` | `doi-bio` | `findVerifiedPapers` |
| `doi` | `doi-bio` | `findVerifiedPapers` |
| `citation` | `doi-bio` | `verifyCitation` |
| `verify` | `doi-bio` | `verifyCitation` |

The exported `RESEARCH_TOOLS` schema enum is updated in lockstep:

```python
"domain": {
    "enum": ["auto", "literature", "trials", "compounds",
             "verified_papers", "citation"],
}
```

### 4. `master_paper_mcp` Gateway Fan-out

A new additive integration in `ThinkingOrchestrator._execute_tools()` invokes `master_paper_mcp.search_papers` in parallel with the normal tool fan-out whenever the active umbrellas indicate it's relevant.

**Umbrella → subjects map** (`MASTER_PAPER_MCP_UMBRELLA_SUBJECTS`):

```python
{
  "ai":  ["ai", "computer", "math", "physics"],
  "bio": ["bio", "clinical", "chemistry"],
  "web": ["social", "other"],
}
```

**Activation** — an umbrella joins the fan-out when either:
- a tool in that umbrella was selected by deliberation (`arxiv`/`github`/`huggingface`→ai, `bio`→bio, `web`→web), **or**
- the deliberation's `intent_weights[umbrella] ≥ 0.15`.

**Health-cache** (`_MASTER_HEALTH_TTL_S = 30 s`) — `_master_paper_mcp_healthy()` probes `health_self` with the result cached for 30 seconds. Any failure or exception counts as unhealthy, and the fan-out is silently skipped:

```text
master_paper_mcp_unhealthy   error=...
master_paper_mcp_skipped     umbrella=ai reason=unhealthy
```

**Parallel subject shards** — `_invoke_master_paper_mcp(umbrella, query, limit)` `asyncio.gather`s one `search_papers` call per subject in the umbrella. Non-empty results are returned as `{source: "master_paper_mcp:<subject>", tool: "search_papers", data: ...}` and mix into `tool_results["mcp_results"]` alongside the regular bio/arxiv/etc payloads.

**Never fails the request** — all subject calls catch exceptions and log `master_paper_mcp_subject_failed` / `master_paper_mcp_subject_exception` warnings; the overall request proceeds on existing sources. The index-based error attribution in `_execute_tools` was updated to label master-task errors as `master_paper_mcp:<umbrella>` instead of mis-indexing into `active_tools`.

Configuration lives in `~/.dova.json` as an HTTP MCP (e.g. `http://localhost:8084/mcp`). No change is needed for users who don't run it — the health probe handles absence cleanly.

### 5. `master_paper_mcp` Performance Safeguards

The initial v1.9 integration could spawn up to **9 parallel `search_papers` shards** (ai: 4 + bio: 3 + web: 2) per request, each reaching the MCP client's 45 s timeout. Under real load this produced `MCP error -32001: Request timed out` on `dova_research`. The shard discipline was tightened along four axes without losing coverage:

- **Query-keyword subject ranking** — `_select_master_subjects(umbrella, query)` scores each subject's keyword set against the query text and keeps at most **2 subjects per umbrella**. When no keywords match, a single umbrella-default subject is used (`ai`/`bio`/`social`) instead of fanning out to all shards. Keyword tables live in `MASTER_PAPER_MCP_SUBJECT_KEYWORDS` (`config/mcp_servers.py`).
- **Global per-request cap** — `_MASTER_MAX_SUBJECTS_PER_REQUEST = 3` bounds total shards across all active umbrellas.
- **Tighter umbrella gate** — the intent-weight activation threshold is raised from `≥ 0.15` to `≥ 0.25` so the 10 % web floor (v1.7) no longer drags `master_paper_mcp:web` into pure-AI or pure-bio queries. Explicit tool selection still activates an umbrella regardless of weight.
- **Per-shard hard timeout** — each `search_papers` call is wrapped in `asyncio.wait_for(..., timeout=_MASTER_SUBJECT_TIMEOUT_S=20s)`. One stalled downstream DB can no longer stretch the overall request.
- **Per-subject failure cache** — shards that errored, timed out, or returned `success=False` are recorded in `_master_mcp_subject_failures` and skipped for `_MASTER_SUBJECT_FAILURE_TTL_S = 120 s`. Repeated slow failures stop charging against the request's critical path.

New log events:

```text
master_paper_mcp_subject_timeout  subject=physics timeout_s=20.0
```

### 6. `doi-bio` Plumbed Through Existing Synthesis Paths

Three downstream consumers in `thinking_orchestrator.py` were extended to recognise `doi-bio` alongside `pubmed-bio` / `clinicaltrials-bio` / `pubchem-bio`:

- **Cross-domain bridge gating** (`_analyze_cross_domain`) — `has_bio` now also checks for `doi-bio` results when deciding whether to run the bridge step.
- **Bio block rendering** — snippets from `doi-bio` flow into the bio section of the synthesis prompt.
- **Bio char-budget** — `doi-bio` entries share the elevated `bio_budget_chars` allocation (v1.8's raised cap), so hydrated DOI/abstract payloads pass through synthesis intact.

---

## Observability — New Log Events

```text
master_paper_mcp_fanout_starting  umbrella=ai subjects=[...] query=...
master_paper_mcp_unhealthy        error=...
master_paper_mcp_skipped          umbrella=ai reason=unhealthy
master_paper_mcp_subject_failed   subject=ai error=...
master_paper_mcp_subject_exception subject=bio error=...
```

All other events (`executing_bio_fanout`, `cross_domain_bridges`, `drug_story_chained`, etc.) are unchanged.

---

## Tests

New / updated unit coverage in `tests/unit/agents/test_bio_routing.py`:

- `test_bio_mcp_servers_list_matches_configs` — canonical list now includes `doi-bio`.
- `test_doi_bio_is_stdio` — validates transport, `npx` command, and that `findVerifiedPapers` + `verifyCitation` tools are present.
- `test_bio_keyword_routing` — new parametrised cases for DOI / Semantic Scholar / OpenAlex / "verify citation" queries.
- `test_bio_fanout_semantic` — cross-DB query hits `{doi-bio, pubmed-bio}`.
- `test_mcp_tool_for_query_bio` — `doi-bio` resolves to `findVerifiedPapers`.
- `test_mcp_tool_params_doi_bio` — `{query, source: all, limit: 10}` shape.
- `test_search_bio_tool_dispatch` — all four new `domain` values (`verified_papers`, `doi`, `citation`, `verify`) dispatch correctly.

---

## Files Changed

```
src/dova/config/mcp_servers.py               (BIO_DOI_MCP config + BIO_MCP_SERVERS roster + master_paper_mcp constants)
src/dova/agents/thinking_orchestrator.py     (master_paper_mcp integration + health cache + doi-bio routing & plumbing)
src/dova/tools/research_tools.py             (search_bio_tool: +verified_papers/doi/citation/verify domains)
tests/unit/agents/test_bio_routing.py        (doi-bio routing, params, dispatch coverage)
docs/release_notes_v1.9.md                   (this file)
README.md                                    (v1.9 tagline + new highlights)
pyproject.toml                               (1.8.0 → 1.9.0)
src/dova/__init__.py                         (__version__ = 1.9.0)
```

---

## Upgrade Notes

- **No breaking API changes.** Response schemas (`ResearchResponse`, `ChatResponse`) are unchanged — the new sources contribute into the existing `tool_results["mcp_results"]` stream, which flows into the same downstream bridge / top-papers / synthesis paths.
- **No new environment variables.** `doi-bio` runs via `npx -y github:tfscharff/doi-mcp` on demand; `master_paper_mcp` is configured in `~/.dova.json` if you run it.
- **Node / `npx` required for `doi-bio`.** The first query triggers an `npx` fetch; subsequent calls reuse the cached package. If `npx` is unavailable, `doi-bio` calls error out and the remaining bio servers continue serving the query.
- **`master_paper_mcp` is optional.** When absent, the health probe returns `unhealthy` within 30 s and fan-outs are silently skipped — no user-visible change from v1.8.
- **No migration required.** No database schemas, MCP configs (other than the opt-in `master_paper_mcp` HTTP entry), or session formats were altered.
