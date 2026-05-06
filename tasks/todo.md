# Add doi-mcp (tfscharff/doi-mcp) to the `bio` umbrella

## Goal
Register `doi-mcp` as a new bio MCP source (`doi-bio`) under the existing `bio`
umbrella so DOI/citation/multi-DB academic queries route to it, and so the
orchestrator can verify citations and fetch verified papers across 9 DBs
(CrossRef, OpenAlex, PubMed, Semantic Scholar, DBLP, zbMATH, ERIC, HAL,
INSPIRE-HEP).

## Tools exposed by doi-mcp
- `verifyCitation(title?, authors?, year?, doi?, journal?)` → verified flag + full citation
- `findVerifiedPapers(query, source="all"|crossref|openalex|pubmed|zbmath|eric|hal|inspirehep|semanticscholar|dblp, limit=5, yearFrom?, yearTo?)`

Transport: STDIO, `npx -y github:tfscharff/doi-mcp`. Zero-config (no API keys).

## Design decisions
- **Peer bio server** (option A) — routed via DOI/citation keywords in
  `_select_bio_servers` / `_select_best_mcp_server`. Start simple; promote
  to synthesis-time citation verifier (option B) later.
- **Primary tool** = `findVerifiedPapers` (search). `verifyCitation` exposed
  via the `domain="citation"` path of `search_bio_tool`.
- **No changes to `_BIO_INTENT_KEYWORDS`** — adding generic tokens like
  `doi`/`citation` would over-trigger the bio umbrella for arxiv-style queries.
  Let `doi-bio` only compete *within* the bio umbrella once bio is chosen.

## Plan

- [ ] 1. `src/dova/config/mcp_servers.py`
      - [ ] Define `BIO_DOI_MCP = MCPServerConfig(name="doi-bio", transport=STDIO,
            command="npx -y github:tfscharff/doi-mcp", priority=2, tools=[...])`
            with `findVerifiedPapers` + `verifyCitation` tool schemas.
      - [ ] Append `"doi-bio"` to `BIO_MCP_SERVERS`.
      - [ ] Register in `get_default_registry()` alongside other bio configs.
      → verify: `from dova.config.mcp_servers import get_default_registry; reg = get_default_registry(); assert "doi-bio" in reg.servers`

- [ ] 2. `src/dova/tools/research_tools.py` — `search_bio_tool`
      - [ ] Extend `domain_map` with `"citation"/"doi"/"verify"` → `("doi-bio", "verifyCitation")`
            and `"verified_papers"/"crossref"/"openalex"/"semantic scholar"` → `("doi-bio", "findVerifiedPapers")`.
      - [ ] Per-server param builder: `findVerifiedPapers` takes `{query, limit, source}`;
            `verifyCitation` takes `{title}` (best-effort mapping from a free-text query).
      - [ ] Extend the `search_bio` tool schema `domain` enum to include
            `"citation"` and `"verified_papers"`.
      → verify: `search_bio_tool("aspirin safety", domain="verified_papers")["mcp_server"] == "doi-bio"`

- [ ] 3. `src/dova/agents/thinking_orchestrator.py`
      - [ ] Add `"doi-bio"` keywords in `_select_bio_servers` (L1911):
            `doi`, `crossref`, `openalex`, `semantic scholar`, `dblp`, `zbmath`,
            `inspirehep`, `eric`, `hal`, `verify citation`, `verified paper`,
            `citation check`.
      - [ ] Add same keywords to `_select_best_mcp_server` bio block (L2013).
      - [ ] Add `"doi-bio": "findVerifiedPapers"` to tool_mapping in
            `_get_mcp_tool_for_query` (L2069).
      - [ ] Add param branch in `_get_mcp_tool_params` (L2125):
            `return {"query": query, "limit": 10, "source": "all"}` for doi-bio.
      → verify: unit tests cover DOI keyword → doi-bio selection

- [ ] 4. Source-tag branches in `thinking_orchestrator.py`
      - [ ] L1520 bridge `has_bio` check: include `"doi-bio"` alongside
            clinicaltrials/pubchem.
      - [ ] L1556 bridge summary loop: include `"doi-bio"`.
      - [ ] L3301 drug-story extractor: either ignore `doi-bio` (safe default)
            or add a `doi_blob` branch if time permits. **Start with ignore.**
      → verify: bridge prompt includes doi-bio snippets when returned.

- [ ] 5. Tests — `tests/unit/agents/test_bio_routing.py`
      - [ ] `BIO_MCP_SERVERS` membership now includes `doi-bio`.
      - [ ] `search_bio_tool("verify this paper", domain="citation")` routes to
            `doi-bio` + `verifyCitation`.
      - [ ] `_select_bio_servers` picks `doi-bio` for queries like
            "crossref DOI lookup for ...", "semantic scholar citations for ...".
      - [ ] `_select_best_mcp_server([...bio_servers], "verify DOI 10.xxx/yyy")`
            returns `doi-bio`.
      - [ ] `_get_mcp_tool_params("doi-bio", "findVerifiedPapers", q)` returns
            `{query, limit, source}`.
      → verify: `pytest tests/unit/agents/test_bio_routing.py -q`

- [ ] 6. Regression sweep
      - [ ] `pytest tests/unit -q` passes (249 tests baseline).
      → verify: no failures; no test marks `doi-bio` as unexpected.

## Non-goals (deferred)
- Synthesis-time `verifyCitation` hook on emitted bibliography (option B).
- Auto-promoting general "papers about X" queries to `doi-bio` vs arxiv.
- HTTP-wrapper hosting of doi-mcp.

## Review
- Files touched:
  - `src/dova/config/mcp_servers.py` — `BIO_DOI_MCP` + `BIO_MCP_SERVERS` + registration
  - `src/dova/tools/research_tools.py` — `search_bio_tool` domain map + schema enum
  - `src/dova/agents/thinking_orchestrator.py` — `_select_bio_servers`,
    `_select_best_mcp_server`, `_get_mcp_tool_for_query`, `_get_mcp_tool_params`,
    bridge/source-tag branches (L1520, L1556, L3000).
  - `tests/unit/agents/test_bio_routing.py` — 5 new tests + parametrize extensions.
- Test delta: 53 → 53 parametrized slots (added 9 new cases total); full unit
  suite 293 passed (no regressions).
- Known limitations:
  - doi-mcp is STDIO/npx — host needs Node. No auto-skip if npx missing;
    connection will fail at runtime with a clear error.
  - `_BIO_INTENT_KEYWORDS` not modified — DOI queries won't themselves raise
    bio-intent weight; doi-bio only activates once bio is chosen by other signals.
  - Drug-story extractor ignores doi-bio payloads (by design; literature is
    already captured in pubmed-bio).
  - `verifyCitation` only exposed via `search_bio_tool(domain="citation")`;
    not yet wired into synthesis-time citation auditing (option B, deferred).
