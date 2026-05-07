# DOVA v2.0 Release Notes

**Release Date:** May 7, 2026

## Overview

DOVA v2.0 is a **refinement release** focused on making the v1.9
`master_paper_mcp` integration more efficient and more honestly documented.
The headline change is **intent-weighted per-umbrella fan-out**: shard budgets
are now split across active umbrellas proportional to the query's semantic
intent, so low-weight umbrellas stop burning downstream MCP capacity on
results that synthesis would trim anyway.

Nothing in the public API, response schema, or on-disk config changes. v1.9
deployments upgrade in place.

---

## Highlights

| Area | v1.9 behaviour | v2.0 behaviour |
|------|----------------|----------------|
| `master_paper_mcp` shard `limit` | Uniform `limit` per umbrella regardless of intent | **Split proportional to `intent_weights`** with floor of 1 per active umbrella |
| `intent_weights` docstring | Claimed synthesis-only use | **Accurately documents both** synthesis weighting **and** master_paper_mcp activation gating |
| `_umbrella_by_tool` comment | Terse, didn't explain omissions | **Explicit rationale** — covers every name `get_available_tools` surfaces; `image` / `awslabs` intentionally absent |

---

## What Changed

### 1. Intent-Weighted Per-Umbrella Shard Limit

**File:** `src/dova/agents/thinking_orchestrator.py` (`_collect_results`,
~line 1538).

When `master_paper_mcp` fans out across the `ai` / `bio` / `web` umbrellas,
v1.9 passed the same `limit` to every umbrella. For a query scored
`{ai: 0.80, bio: 0.10, web: 0.10}`, `master_paper_mcp:bio` and
`master_paper_mcp:web` still requested the full limit — only to have
`_allocate_top_papers` trim those results during synthesis. That wasted both
downstream MCP calls and wall-clock budget on shards whose output would
never reach the final `top_papers` slate.

v2.0 introduces `_limit_for(umbrella)` inside `_collect_results`:

```python
_intent_weights = deliberation.intent_weights or {}
_active_weight_total = sum(
    max(0.0, float(_intent_weights.get(u, 0.0))) for u in umbrella_subjects
)

def _limit_for(umbrella: str) -> int:
    if _active_weight_total > 0:
        w = max(0.0, float(_intent_weights.get(umbrella, 0.0)))
        share = w / _active_weight_total
    else:
        share = 1.0 / max(len(umbrella_subjects), 1)
    return max(1, round(share * limit))
```

**Properties:**

- **Proportional allocation.** With `limit=10` and weights
  `{ai: 0.7, bio: 0.2, web: 0.1}`, umbrellas receive `{ai: 7, bio: 2, web: 1}`
  shard capacity.
- **Floor of 1.** If an umbrella was activated (via user-selected tool or
  intent weight ≥ 0.25) but scored tiny, it still gets at least 1 slot so
  the activation decision isn't silently reversed.
- **Zero-weight fallback.** When all active umbrellas have zero weight —
  which can happen if the user forced activation via explicit tool
  selection — the split falls back to uniform across active umbrellas.
- **Per-subject limit unchanged.** Splitting happens only at the umbrella
  level. The sub-subject cap inside `_invoke_master_paper_mcp` still uses
  whatever limit the umbrella received, because fragmenting further would
  make shard rankings unreliable.

**Worked examples** (`limit=10`):

| `intent_weights` | Result |
|------------------|--------|
| `{ai: 0.70, bio: 0.20, web: 0.10}` | `{ai: 7, bio: 2, web: 1}` |
| `{ai: 0.10, bio: 0.80, web: 0.10}` | `{ai: 1, bio: 8, web: 1}` |
| `{ai: 1.00}` (single umbrella) | `{ai: 10}` |
| `{ai: 0.95, bio: 0.04, web: 0.01}` | `{ai: 10, bio: 1, web: 1}` (floors hold) |
| All-zero active weights | Uniform across active umbrellas |

Rounding can produce a small under-budget sum (e.g., `{0.34, 0.33, 0.33} ×
10 = {3, 3, 3}`, total 9). This is intentional — `limit` is a per-shard
ceiling, not a hard-total budget, and the alternative (floor instead of
round) would reliably over-spend.

### 2. `intent_weights` Docstring Corrected

**File:** `src/dova/agents/thinking_orchestrator.py` (`Deliberation`
dataclass, ~line 383).

v1.9's field comment claimed intent weights were used *only* at synthesis
time, not execution. In practice they've also gated `master_paper_mcp`
umbrella fan-out since v1.9 (activation threshold `≥ 0.25` at the execution
path). v2.0 updates the comment to reflect both uses — now readers of the
dataclass know the field drives both ranking *and* routing:

```python
# Semantic intent weights across {ai, bio, web}. Sum to 1.0.
# Used downstream (a) to weight result aggregation in synthesis, and
# (b) to gate master_paper_mcp umbrella fan-out in _collect_results
# when a weight meets the activation threshold.
intent_weights: dict[str, float] = field(default_factory=dict)
```

### 3. `_umbrella_by_tool` Rationale Added

**File:** `src/dova/agents/thinking_orchestrator.py` (~line 1477).

The `_umbrella_by_tool` map that routes tool names to umbrella groups lists
`arxiv` / `github` / `huggingface` / `hugging-face` / `bio` / `web` and no
others. Without context a reader might expect `pubmed` / `clinicaltrials` /
`pubchem` / `image` / `awslabs` — v2.0 adds a comment explaining why they're
intentionally absent:

- The deliberation LLM only ever sees umbrella-level tool names
  (`get_available_tools` at line ~63 aggregates bio sub-servers behind the
  `bio` umbrella name). So `bio` transparently covers `pubmed-bio` /
  `clinicaltrials-bio` / `pubchem-bio` / `doi-bio`.
- `image` and `awslabs` aren't paper sources, so paper fan-out intentionally
  skips them.

---

## Verification

The v2.0 changes were validated against the full test matrix with no
regressions:

| Suite | Count | Runtime | Status |
|-------|-------|---------|--------|
| Unit tests | 298 / 298 | 2.0 s | ✅ |
| Integration tests | 24 / 24 | 37.0 s | ✅ |
| ruff delta on `thinking_orchestrator.py` | +0 | — | ✅ |
| mypy delta on `thinking_orchestrator.py` | +0 | — | ✅ |

The weighted-limit math was additionally verified against 6 cases spanning
AI-dominant, bio-dominant, single-umbrella, all-zero fallback, tiny-floor,
and round-drift scenarios.

---

## Files Changed

```
pyproject.toml                               (1.9.0 → 2.0.0)
src/dova/__init__.py                         (__version__ = 2.0.0)
src/dova/config/settings.py                  (app_version default → 2.0.0)
src/dova/agents/thinking_orchestrator.py     (intent-weighted _limit_for + docstring + comments)
docs/release_notes_v2.0.md                   (this file)
README.md                                    (v2.0 tagline + highlight)
```

---

## Upgrade Notes

- **No breaking API changes.** Response schemas, MCP configs, and session
  formats are all unchanged. `ResearchResponse.top_papers` still
  post-filters the per-umbrella fan-out — the v2.0 change affects only how
  much is fetched *before* that filter.
- **No new environment variables.** The weighted-limit behaviour activates
  automatically for any request whose `deliberation.intent_weights` is
  non-empty, which is every request on the thinking path.
- **Expected production impact.** For ai-heavy or bio-heavy queries
  (weights skewed ≥ 0.7 on one axis) the less-relevant umbrellas now fetch
  ~10–20 % as many shard results as before. Wall-clock savings scale with
  how much the downstream MCPs were the bottleneck.
- **No migration required.** No schemas, MCP configs, or session formats
  were altered.

---

## Prior Releases

- [v1.9](release_notes_v1.9.md) — `doi-bio` MCP (9 DBs) + `master_paper_mcp` gateway
- [v1.8](release_notes_v1.8.md) — Cross-domain AI ⇄ Bio analyst, drug-story chain, env-driven LLM config
- [v1.7](release_notes_v1.7.md) — Weighted intent deliberation, grouped source selector
- [v1.6](release_notes_v1.6.md), [v1.5](release_notes_v1.5.md), [v1.4](release_notes_v1.4.md), [v1.3](release_notes_v1.3.md)
