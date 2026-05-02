# DOVA v1.8 Release Notes

**Release Date:** May 2, 2026

## Overview

DOVA v1.8 makes the orchestrator **operationally reliable on bio-heavy queries** and adds **cross-field reasoning** so AI findings and biomedical evidence are explicitly bridged rather than presented side-by-side. Under the hood, every LLM configuration is now driven entirely from `.env` — no hardcoded model IDs remain in the source tree.

The v1.8 cycle was driven by a single user-visible complaint: *"Dova reports AI content even for bio-heavy queries."* Diagnosing that exposed three compounding bugs in the bio path and a missing cross-pollination step in synthesis. This release fixes all four.

---

## Highlights

| Area | v1.7 behaviour | v1.8 behaviour |
|------|----------------|----------------|
| Bio fan-out | PubMed excluded whenever narrow keyword list missed | **PubMed always joins the fan-out** when bio is invoked (literature is nearly always relevant) |
| PubMed query | Raw natural-language query (often 0 hits) | **`_distill_pubmed_query()`** strips stopwords/years/filler → 455+ hits where 0 were returned before |
| PubMed payload | Bare PMID list (~450 chars, no titles) | **Two-step hydrate** via `pubmed_fetch_articles` → full titles, abstracts, MeSH terms (~48KB) |
| Cross-field reasoning | AI and Bio results listed separately | **Cross-domain analyst** produces explicit `{ai_method, bio_target, mechanism, novelty, feasibility, testable_prediction}` bridges |
| Bio → AI analogies | Not surfaced | **Curated reframe map** (24 pairings: immune, olfactory, predictive coding, dopamine, hippocampus, …) injected into synthesis prompt |
| Drug story coherence | PubChem + PubMed + ClinicalTrials as 3 disjoint blobs | **`_extract_drug_story()`** stitches compound → mechanism → trial into one structured unit |
| `dova mcp serve` | Missing `top_papers`/`pubmed_papers` in output | Parity with `/api/v1/research`: same deliberation-first path, same fields |
| Web UI paper URLs | `top_papers` derived by `/api/v1/research` but not chat endpoints → UI fell back to arxiv-only | Chat path now routes through `extract_research_data()` → UI shows weighted AI/Bio top-10 |
| LLM configuration | Mix of hardcoded literals and partial env overrides | **100 % env-driven** — every model ID, embedding, provider-priority reads `.env` |
| MCP serve HTTP | DNS-rebinding protection blocked external Host headers (421) | `--allowed-host`, `--allowed-origin`, `--no-dns-rebinding-protection` flags |

---

## New Features

### 1. Cross-Domain AI ⇄ Bio Analyst (Axis 1)

A new `_analyze_cross_domain(query, deliberation, tool_results)` step runs after tool fan-out and before synthesis. It passes the top AI evidence (arxiv papers, GitHub repos, HuggingFace models) alongside the top Bio evidence (PubMed, ClinicalTrials, PubChem) to a dedicated LLM prompt that emits **structured bridges**:

```json
{
  "ai_method": "Equivariant GNN (arxiv.org/abs/X)",
  "bio_target": "Protein-protein interaction prediction (pubmed.ncbi.nlm.nih.gov/Y/)",
  "mechanism": "GNNs naturally encode relational structure of PPI networks.",
  "novelty": 0.6,
  "feasibility": 0.75,
  "testable_prediction": "Benchmark against the STRING database with AUROC > 0.85."
}
```

**Gating (no cost for out-of-scope queries):**
- Both `ai_weight ≥ 0.1` AND `bio_weight ≥ 0.1`
- At least 1 AI item AND 1 bio signal present
- Pure-AI and pure-bio queries skip the step entirely

**Surfacing:** bridges are rendered as a dedicated "Cross-Domain Bridges" section in the synthesis prompt, with explicit instructions to weave ≥ 2 bridges into the answer. They're exposed as `research_results.cross_domain_bridges` in the API and as `🔗 Cross-Domain Bridges` cards in the web UI.

Cost: one extra LLM call (`TaskType.REASONING`, ~3000 tokens). Skipped when gated.

### 2. Bio → AI Reframe Injection (Axis 2)

A curated keyword → analogue map (`_BIO_TO_AI_REFRAMES`, 24 entries) injects a one-line hint into the synthesis prompt when the query uses biological vocabulary and the AI group has ≥ 0.3 weight. Examples:

```
- olfactory → sparse distributed representations, mixture-of-experts routing, LSH
- immune    → clonal selection, negative selection, affinity maturation, adversarial critics
- hippocampus → episodic memory buffer, retrieval-augmented models, episodic control
- predictive coding → hierarchical top-down generative models, free-energy formulations
```

Zero LLM cost. At most 3 analogues injected per query to keep the prompt tight.

### 3. Drug Story Chaining (Axis 3)

`_extract_drug_story(mcp_results, query)` stitches PubChem + PubMed + ClinicalTrials payloads into a single coherent structure:

```json
{
  "compound": "Semaglutide",
  "pubchem_cid": "56843331",
  "pubchem_url": "https://pubchem.ncbi.nlm.nih.gov/compound/56843331",
  "mechanism_pmids": ["38843460", "39114288", ...],
  "mechanism_pmid_urls": ["https://pubmed.ncbi.nlm.nih.gov/38843460/", ...],
  "trial_nct_ids": ["NCT05869903", ...],
  "trial_urls": ["https://clinicaltrials.gov/study/NCT05869903", ...],
  "summary": "Drug story for Semaglutide; (PubChem CID 56843331); literature: 5 articles; trials: 3 studies."
}
```

Pure string/dict processing — no LLM call. Rendered as a dedicated "Drug Story (PubChem → PubMed → ClinicalTrials)" block in the synthesis prompt and as a `💊 Drug Story` card in the web UI.

### 4. Bio-Flow Reliability Fixes

Three compounding bugs were uncovered while debugging *"bio-heavy queries return AI content only"*:

**Bug A — PubMed skipped by selector.** `_select_bio_servers()` required a narrow keyword match before including `pubmed-bio` in the fan-out, so queries like `"GLP-1 receptor agonists efficacy"` hit only `clinicaltrials-bio`. Fix: PubMed is now **always** included when the bio umbrella is invoked; literature is universally relevant. (`thinking_orchestrator.py:_select_bio_servers`)

**Bug B — 0 PubMed hits on natural-language queries.** The raw user query was passed to PubMed verbatim. NCBI's parser ANDs every token, so `"latest clinical trials for GLP-1 receptor agonists in obesity treatment 2025, safety profile and efficacy 2026"` returned **0** results. The new `_distill_pubmed_query()` strips stopwords, filler phrases, years, and punctuation — the same distilled query returns **455** results. (`thinking_orchestrator.py:_distill_pubmed_query`, `_get_mcp_tool_params`)

**Bug C — PubMed search returns only PMIDs.** The hosted MCP returns `**PMIDs:** 38843460, 39114288, ...` markdown with no titles or abstracts, so the synthesis LLM had no substance to reason about. A **second-pass hydration** calls `pubmed_fetch_articles` to expand PMIDs into full articles (title + authors + abstract + MeSH terms, ~48KB). Synthesis prompt char-cap for bio was also raised from 2,500 × bio_weight to 12,000 × bio_weight so the hydrated content passes through mostly intact. (`thinking_orchestrator.py:_execute_mcp_tool` post-fetch block)

### 5. Top-N Paper URLs with AI/Bio Split

`ThinkingOrchestrator.extract_research_data()` now returns `top_papers` — up to **10 paper URLs** allocated proportionally to the query's intent weights:

- `ai=0.7, bio=0.3` → 7 arxiv + 3 pubmed
- `ai=0.05, bio=0.85` → 1 arxiv + 9 pubmed
- `ai=1.0, bio=0.0` → 10 arxiv + 0 pubmed

Splits are computed by `_allocate_top_papers()`. Hard-zero weights are respected (`bio=0` → zero pubmed slots, not forced minimum). `_extract_pubmed_papers()` handles three payload shapes — nested JSON (`{articles: [...]}`), hydrated markdown blocks (`### Title ... **PMID:**`), and bare PMID lists.

Web UI cards (🔗 Papers in both `buildResearchAndDebateCards` and `addAssistantMessage`) prefer `top_papers` with a `[PubMed]` badge on pubmed rows, fall back to arxiv-only `papers` when not available.

### 6. `dova mcp serve` Parity with `dova serve`

The MCP server (`src/dova/mcp_server.py`) previously ran a thin `ResearchAgent` with no orchestrator, no memory, no source registry, and no `setup_mcp_repos()` — so its `dova_research` tool returned a much thinner payload than `/api/v1/research`. v1.8 wires the MCP server through the **same** `ThinkingOrchestrator` with the **same** supporting services as `api/main.lifespan`:

- `EnhancedMemoryService` (wired to the LLM router for embeddings)
- `AgentCoreMemoryService` (when AWS is configured)
- `SourceRegistry`
- `ResearchAgent` with `source_registry` + `enhanced_memory_service` + `tavily_api_key`
- `DebateAgent` (2 rounds)
- `setup_mcp_repos()` (ensures arxiv MCP is cloned/updated)

`dova_research` MCP tool now returns the full payload including `top_papers`, `pubmed_papers`, `cross_domain_bridges`, and `drug_story` — byte-for-byte consistent with `/api/v1/research`.

**Source-group support:** `dova_research(sources="ai,bio")` is now accepted. The MCP tool expands group tokens (`ai`, `web`, `bio`) to concrete sources using the same `SOURCE_GROUP_MAP` as the web UI. `sources="all"` maps to `["arxiv", "github", "huggingface", "web", "bio"]` (previously dropped `bio`, now included).

### 7. 100% `.env`-Driven LLM Configuration

Every hardcoded model ID in the serve paths has been replaced with an `os.environ.get(..., default)` lookup. No model ID, embedding ID, or provider priority is baked into source any more.

**Fixed locations:**

| file:line | before | after |
|---|---|---|
| `providers.py:70-75` | literal Anthropic tier dict | `ANTHROPIC_MODEL_{BASIC,STANDARD,ADVANCED,REASONING}` |
| `providers.py:77-82` | literal OpenAI tier dict | `OPENAI_MODEL_{BASIC,STANDARD,ADVANCED,REASONING}` |
| `providers.py:86-89` | literal OpenAI token cap dict | `OPENAI_MAX_TOKENS_GPT_5_4[_MINI]`, `OPENAI_DEFAULT_MAX_TOKENS` |
| `providers.py:708` | inline Bedrock embedding default | `BEDROCK_EMBEDDING_MODEL` env |
| `providers.py:813` | hardcoded `text-embedding-3-small` | `OPENAI_EMBEDDING_MODEL` env |
| `providers.py:716/768/856` | `priority=1/2/3` literal | `_provider_priority()` driven by `LLM_PROVIDER_ORDER` |
| `settings.py:22` | `bedrock_model_id` literal default | reads `BEDROCK_MODEL_STANDARD` at init |
| `agents/base.py:441` | `model_id` param literal default | `None` + env fallback at call site |

**New `.env` variables** (see `.env.example` for the full template):

```bash
# Provider fallback order (primary first)
LLM_PROVIDER_ORDER=bedrock,anthropic,openai

# Bedrock tiers
BEDROCK_MODEL_BASIC=...
BEDROCK_MODEL_STANDARD=...
BEDROCK_MODEL_ADVANCED=...
BEDROCK_MODEL_REASONING=...
BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0

# Anthropic tiers (secondary)
ANTHROPIC_MODEL_BASIC=...
ANTHROPIC_MODEL_STANDARD=...
ANTHROPIC_MODEL_ADVANCED=...
ANTHROPIC_MODEL_REASONING=...

# OpenAI tiers (tertiary) + output caps
OPENAI_MODEL_BASIC=...
OPENAI_MODEL_STANDARD=...
OPENAI_MODEL_ADVANCED=...
OPENAI_MODEL_REASONING=...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_DEFAULT_MAX_TOKENS=16384
OPENAI_MAX_TOKENS_GPT_5_4=16384
OPENAI_MAX_TOKENS_GPT_5_4_MINI=16384
```

### 8. `dova mcp serve` HTTP Hardening

The MCP SDK's `TransportSecuritySettings` defaults to DNS-rebinding protection with allowed hosts `localhost`, `127.0.0.1`, `[::1]`. External HTTP clients (e.g. `mcp1.cavatar.info:8083`) got `421 Misdirected Request`. Three new CLI flags:

```bash
dova mcp serve --transport http --port 8083 \
  --allowed-host 'mcp1.cavatar.info:*' \
  --allowed-origin 'https://mcp1.cavatar.info' \
  --no-dns-rebinding-protection   # escape hatch for trusted networks
```

### 9. STDIO Buffer Limit — Fix for ArXiv `Separator is not found`

The arxiv MCP server returns a single-line JSON-RPC response per tool call. With 10+ full paper abstracts in one response, that line routinely exceeds asyncio's default `StreamReader` buffer of **64 KiB**, tripping:

```text
retry_attempt     error='Separator is not found, and chunk exceed the limit'
retry_exhausted   attempts=4
mcp_invoke_failed server=arxiv tool=search_papers
```

Retries couldn't recover — the same oversized line came back each time.

**Fix:** `_invoke_stdio` now spawns the subprocess with `limit=10 * 1024 * 1024` (10 MiB), covering any realistic MCP response. Standard Python idiom for subprocess `StreamReader`s that handle large lines. (`tools/mcp_registry.py:_invoke_stdio`)

The related `stdio_timeout=60` log line (cold `uv` resolve + search taking >60s) is left at 60s intentionally — one-off slowness that retries recover from.

---

## API Changes

### `ResearchResponse` (`POST /api/v1/research`, `/chat/stream`)

New fields (all additive, defaults to empty):

```json
{
  "top_papers": [
    {"title": "...", "url": "...", "source": "arxiv" | "pubmed", "description": "..."}
  ],
  "pubmed_papers": [
    {"title": "...", "url": "...", "metadata": {"pmid": "...", "authors": [...]}}
  ],
  "cross_domain_bridges": [
    {
      "ai_method": "...", "bio_target": "...", "mechanism": "...",
      "novelty": 0.6, "feasibility": 0.75, "testable_prediction": "..."
    }
  ],
  "drug_story": {
    "compound": "...", "pubchem_cid": "...", "pubchem_url": "...",
    "mechanism_pmids": [...], "mechanism_pmid_urls": [...],
    "trial_nct_ids": [...], "trial_urls": [...],
    "summary": "..."
  }
}
```

### `ResearchResponse.metadata.intent_weights`

`intent_weights` is now mirrored into `ResearchResponse.metadata` (in addition to the SSE `stage` and `complete` events from v1.7) so non-streaming clients don't have to reach into SSE to know the routing split.

### `ChatResponse.research_results`

`research_results` on chat endpoints now includes `top_papers`, `pubmed_papers`, `cross_domain_bridges`, and `drug_story`. Previously these were only on `/api/v1/research`.

### `dova_research` MCP tool

Response payload extended to match `/api/v1/research` (see above). `sources` argument now accepts group tokens (`ai`, `web`, `bio`) in addition to concrete names. `sources="all"` expands to all five sources including `bio`.

---

## Observability

New log events:

```text
pubmed_hydrated       pmids=10 data='48508 chars'
cross_domain_bridges  count=3 ai_weight=0.6 bio_weight=0.3
drug_story_chained    compound=Semaglutide pmids=5 ncts=3
cross_domain_analysis_failed  error=...
```

Existing events (`executing_bio_fanout`, `bio_server_call_complete`, `intent_weights_computed`) are unchanged.

---

## UI Changes

- **🔗 Cross-Domain Bridges card** — new section in the research card showing up to 4 bridges with novelty/feasibility scores and the testable prediction.
- **💊 Drug Story card** — clickable PubChem CID + PMID + NCT links when PubChem was invoked.
- **Papers card** — now uses `top_papers` (weighted AI/Bio split) with a `[PubMed]` badge. Total count shows `(arxiv + pubmed)`.

Both `buildResearchAndDebateCards` and `addAssistantMessage` render the new blocks.

---

## Token Budgets

Targeted bumps (see `docs/token_budget.md` for full reference):

- `agents/thinking_orchestrator.py:_analyze_cross_domain` → `max_tokens` **1,200 → 3,000** (bridge JSON was truncating mid-array)
- `agents/synthesis.py:_generate_emergent_insights` → `max_tokens` **15,000 → 24,000** (headroom for richer emergent-insight JSON)
- `agents/thinking_orchestrator.py:_build_synthesis_prompt` → bio char budget per MCP result raised from `max(400, 2500 × bio_weight)` to `max(1500, 12000 × bio_weight)` so hydrated PubMed passes through

All other `max_tokens` values unchanged (provider ceilings already clamp them).

---

## Tests

- **277 unit tests passing** (up from 267 in v1.7).
- New/updated:
  - `test_bio_routing.py::test_bio_fanout_semantic` — updated to reflect *PubMed is always included* contract (bug A fix).
  - Cross-domain analyst unit tests (gating on ai/bio weights, parse fallback, happy path).
  - Drug-story chain unit tests (PubChem/PubMed/ClinicalTrials extraction, empty-case).
  - `_extract_pubmed_papers` — three payload shapes: hydrated markdown, bare PMID list, inline PMID mentions.
  - `_distill_pubmed_query` — known stopword/filler stripping.

```bash
$ pytest tests/unit -q
... 277 passed in 4.48s
```

---

## Files Changed

```
src/dova/agents/thinking_orchestrator.py     (all three Axis 1 layers + bio fixes + PubMed distiller + hydrate)
src/dova/agents/synthesis.py                 (sub-synthesis token bump)
src/dova/agents/base.py                      (strands helper env-driven default)
src/dova/api/routes/research.py              (top_papers/pubmed_papers/bridges/drug_story on response + metadata.intent_weights)
src/dova/api/routes/chat.py                  (extract_research_data path parity → UI gets weighted paper URLs)
src/dova/api/schemas/research.py             (new fields: top_papers, pubmed_papers, cross_domain_bridges, drug_story)
src/dova/api/static/index.html               (bridges + drug-story UI cards, [PubMed] badge)
src/dova/cli/main.py                         (mcp serve --allowed-host / --allowed-origin / --no-dns-rebinding-protection)
src/dova/mcp_server.py                       (ThinkingOrchestrator wiring + source groups + enriched response)
src/dova/tools/mcp_registry.py               (STDIO subprocess buffer limit 64 KiB → 10 MiB)
src/dova/config/providers.py                 (all model IDs + embeddings + priorities now env-driven)
src/dova/config/settings.py                  (AWSSettings.bedrock_model_id env-driven)
tests/unit/agents/test_bio_routing.py        (PubMed-always-included contract update)
.env                                         (secondary Anthropic, tertiary OpenAI tiers + embedding overrides + provider order)
.env.example                                 (documents all new knobs)
docs/token_budget.md                         (new — token cap reference + operator guidance)
docs/release_notes_v1.8.md                   (this file)
README.md                                    (v1.8 tagline + feature highlights)
pyproject.toml                               (1.6.0 → 1.8.0)
src/dova/__init__.py                         (__version__ = 1.8.0)
```

---

## Upgrade Notes

- **No breaking API changes.** `ResearchResponse` fields are additive. `ChatResponse.research_results` is a free-form dict, so the new keys are forward-compatible.
- **Env migration recommended.** If you've been relying on hardcoded Anthropic/OpenAI defaults, they still resolve identically — no action required. To customise, copy `.env.example` → `.env` and edit the tier IDs for the providers you use.
- **MCP serve over HTTP** needs `--allowed-host` for any non-`localhost` client Host header (see flag docs above).
- **Browser cache**: hard-refresh once after upgrading to see the new Cross-Domain Bridges and Drug Story cards.
- **No migration required.** No database schemas, MCP configs, or session formats were altered.
