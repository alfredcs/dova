# Task: Add Bio/Pharma MCP servers to DOVA

## Goal
Extend DOVA's existing arxiv / github / huggingface research tools with a curated set of
biotech/pharma MCP servers. Keep the addition minimal, token-efficient, and focused on
servers that are validated + likely to stay reachable.

## Principles (why we're NOT adding all 80+ from the directory)
- DOVA's orchestrator deliberation prompt already lists every tool for the LLM to pick from.
  Adding 80 bio servers would explode the prompt, blow up tokens, and confuse the model.
- Unofficial repos with <10 stars have high risk of rot and no HTTP endpoint.
- Optimal design: **one aggregated `bio` tool** like `awslabs`, backed by ~4 curated servers
  routed by keyword at execution time (mirrors existing `_select_best_mcp_server` pattern).

## Curated server set (all independently verified 2026-04-29)

| Key | Repo | Stars | Auth | Why it's included |
|---|---|---|---|---|
| `biomcp` | genomoncology/biomcp | 497 | none (optional API keys boost rate limits) | Broadest coverage — 13 entity types: gene/variant/article/trial/drug/disease/pathway/protein/AE/PGx/GWAS/diagnostic/phenotype. Rust binary, `uv tool install biomcp-cli`. MIT. |
| `pubmed-bio` | cyanheads/pubmed-mcp-server | 88 | none (optional `NCBI_API_KEY`) | Focused biomedical literature; richer than biomcp for PubMed-only queries. Apache 2.0. |
| `clinicaltrials-bio` | cyanheads/clinicaltrialsgov-mcp-server | 67 | none (public v2 API) | ClinicalTrials.gov v2 API — for trial-specific queries. |
| `pubchem-bio` | cyanheads/pubchem-mcp-server | 8 | none | Chemical compound lookup — complements biomcp's drug tool with better small-molecule depth. |

Excluded categories and why:
- Augmented-Nature org (23 servers) — JS/STDIO only, most <10 stars, heavy overlap with biomcp.
- JackKuo666 org — literature search overlap with pubmed + arxiv.
- longevity-genie — niche / alpha / requires external API keys (AlphaGenome, Benchling, FutureHouse).
- bio-mcp org — local bioinformatics tools (BLAST/BWA/samtools) needing Conda envs — not suitable for a hosted service.

## Implementation plan

1. **Config**: add a `bio` umbrella in `src/dova/config/mcp_servers.py` documenting the four
   curated servers with install commands. Keep them STDIO (no hosted HTTP option).
   → verify: `python -c "from dova.config.mcp_servers import ..."` imports.

2. **Orchestrator integration**: in `src/dova/agents/thinking_orchestrator.py`:
   - Add `bio` to `TOOL_DESCRIPTIONS` so deliberation prompt knows about it.
   - Extend `_select_best_mcp_server` keyword map with bio-prefixed server routing
     (trial/clinical → clinicaltrials-bio, chem/compound/smiles → pubchem-bio,
     literature/pubmed/article → pubmed-bio, default → biomcp).
   - Extend `_get_mcp_tool_for_query` and `_get_mcp_tool_params` for each of the 4 servers'
     primary search tools.
   → verify: unit test the routing function with representative queries.

3. **Research tool helper** (optional, thin): add a `search_bio_tool` in
   `src/dova/tools/research_tools.py` mirroring `search_arxiv_tool` so the Strands/agent
   layer can call it explicitly. Returns `{tool_name, mcp_server, params}`.
   → verify: import + dispatch.

4. **Docs**:
   - Update `docs/bio-mcp-servers-dir.md` with a short "Integrated into DOVA" section listing the 4 picks.
   - Note in `DOVA.md` / `README.md` that DOVA now supports bio/pharma research via a `bio` source.

5. **QA**:
   - Add unit test `tests/unit/agents/test_bio_routing.py` for keyword routing & tool/params mapping.
   - Run full `pytest tests/unit` to ensure nothing regressed.
   - Dry-run the deliberation path with a mocked LLM returning `tools_to_use: [{"tool":"bio", ...}]`
     to confirm the tool flows through `_execute_mcp_tool`.

## Out of scope
- Installing any of these servers (STDIO install is user responsibility; we only register config).
- Adding the 75+ other servers from the directory.
- Writing hosted HTTP wrappers (none exist for bio MCP yet).

## Review (2026-04-29)

**Implemented (as planned, with mid-implementation fixes)**

- `src/dova/config/mcp_servers.py`: added `BIO_PUBMED_MCP`, `BIO_CLINICALTRIALS_MCP`, `BIO_PUBCHEM_MCP` (all `MCPTransport.HTTP`, cyanheads hosted endpoints, no auth) and a `BIO_MCP_SERVERS` list. Registered in `get_default_registry()` so they load without user `~/.dova.json` edits.
- `src/dova/agents/thinking_orchestrator.py`:
  - Added `"bio"` to `TOOL_DESCRIPTIONS` and made it always appear in `get_available_tools()`.
  - Hid the individual bio server names from the deliberation LLM (umbrella pattern mirrors `awslabs.*`).
  - `get_mcp_servers_for_tool("bio")` returns the 3-server list.
  - Extended `_select_best_mcp_server` with bio keyword map (trial/NCT/phase → clinicaltrials; SMILES/CID/formula → pubchem; MeSH/PMID/systematic review → pubmed; default fallback → pubmed).
  - Extended `_get_mcp_tool_for_query` with pubmed_search_articles / clinicaltrials_search_studies / pubchem_search_compounds.
  - Extended `_get_mcp_tool_params` with **server-specific schemas** (discovered via live `tools/list` probe after the first dry-run revealed my `{query, max_results}` default was wrong):
    - PubMed: `{query, maxResults}` (camelCase)
    - ClinicalTrials: `{query}` (no max_results field)
    - PubChem: `{searchType: "identifier", identifierType: "name", identifiers: [query]}`
- `src/dova/tools/research_tools.py`: added `search_bio_tool(query, domain, max_results)` with the same per-server shape map and registered it in `RESEARCH_TOOLS`.
- `tests/unit/agents/test_bio_routing.py`: new file — 30 tests covering config sanity, tool surfacing, keyword routing (10 query/server pairs), tool+param mapping, and `search_bio_tool` dispatch.
- `docs/bio-mcp-servers-dir.md`: added "⭐ Integrated into DOVA" section at top listing the 4-row table of adopted servers + rationale for excluding the other 80.
- `README.md`: updated tagline + features list to mention biomedical sources.
- `DOVA.md`: added `Biomedical / Pharma` row to the intelligent source selection table.

**What I considered but rejected**
- Adding `genomoncology/biomcp` (497⭐, 13 entity types) — broader coverage but STDIO-only install (`biomcp-cli`), breaks the zero-install UX of the other three.
- Adding Augmented-Nature / longevity-genie / JackKuo666 orgs — 80+ repos, 90%+ with <10⭐ and STDIO-only, overlap heavy with cyanheads trio, would bloat the deliberation prompt without material coverage gain.
- Surfacing `pubmed-bio` / `clinicaltrials-bio` / `pubchem-bio` as first-class tool names to the LLM — rejected in favor of the `bio` umbrella to minimize prompt tokens.

**QA evidence**
- Unit: 255/255 passing (30 new bio tests + 225 existing, no regressions).
- Live HTTP: successful `initialize → tools/call` roundtrip against all three endpoints with real queries (aspirin, BRAF V600E, caffeine → CID 2244).

**Files changed**
```
src/dova/config/mcp_servers.py             (+~130 lines)
src/dova/agents/thinking_orchestrator.py   (+~35 lines)
src/dova/tools/research_tools.py           (+~60 lines)
tests/unit/agents/test_bio_routing.py      (new, ~160 lines)
docs/bio-mcp-servers-dir.md                (+~45 lines)
README.md                                  (2 line tweaks)
DOVA.md                                    (+1 table row)
```
