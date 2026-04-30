"""Unit tests for the bio MCP umbrella routing in ThinkingOrchestrator.

Covers:
- Config registration of the 3 hosted bio endpoints.
- `bio` appears in the deliberation tool list.
- `get_mcp_servers_for_tool("bio")` returns the curated list.
- `_select_best_mcp_server` routes to the right server by keyword.
- `_get_mcp_tool_for_query` + `_get_mcp_tool_params` map correctly.
- The `search_bio_tool` dispatcher honors the `domain` argument.
"""

import pytest
from unittest.mock import MagicMock

from dova.agents.thinking_orchestrator import (
    ThinkingOrchestrator,
    compute_intent_weights,
    get_available_tools,
    get_mcp_servers_for_tool,
)
from dova.config.mcp_servers import (
    BIO_CLINICALTRIALS_MCP,
    BIO_MCP_SERVERS,
    BIO_PUBCHEM_MCP,
    BIO_PUBMED_MCP,
    MCPTransport,
    get_default_registry,
)
from dova.tools.research_tools import search_bio_tool


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_bio_servers_are_http_with_valid_urls():
    for cfg in (BIO_PUBMED_MCP, BIO_CLINICALTRIALS_MCP, BIO_PUBCHEM_MCP):
        assert cfg.transport == MCPTransport.HTTP
        assert cfg.url and cfg.url.startswith("https://")
        assert cfg.enabled is True
        assert cfg.tools, f"{cfg.name} has no tools declared"


def test_bio_mcp_servers_list_matches_configs():
    names = {BIO_PUBMED_MCP.name, BIO_CLINICALTRIALS_MCP.name, BIO_PUBCHEM_MCP.name}
    assert set(BIO_MCP_SERVERS) == names


def test_default_registry_registers_bio_servers():
    registry = get_default_registry()
    for name in BIO_MCP_SERVERS:
        srv = registry.get_server(name)
        assert srv is not None, f"{name} not registered"
        assert srv.enabled


# ---------------------------------------------------------------------------
# Orchestrator-level tool surfacing
# ---------------------------------------------------------------------------

def test_bio_in_available_tools():
    tools = get_available_tools()
    assert "bio" in tools
    # Individual bio server names must be hidden to avoid bloating the
    # deliberation prompt — the umbrella routes internally.
    for name in BIO_MCP_SERVERS:
        assert name not in tools


def test_get_mcp_servers_for_bio():
    assert get_mcp_servers_for_tool("bio") == list(BIO_MCP_SERVERS)


# ---------------------------------------------------------------------------
# Keyword-based sub-routing
# ---------------------------------------------------------------------------

@pytest.fixture
def orch():
    return ThinkingOrchestrator(llm_router=MagicMock(), agents={}, mcp_client=None)


@pytest.mark.parametrize(
    "query,expected_server",
    [
        ("latest BRAF V600E clinical trials for melanoma", "clinicaltrials-bio"),
        ("phase III trial enrollment criteria", "clinicaltrials-bio"),
        ("SMILES structure for aspirin", "pubchem-bio"),
        ("PubChem CID for caffeine", "pubchem-bio"),
        ("molecule similarity search", "pubchem-bio"),
        ("PubMed articles on alpha-synuclein aggregation", "pubmed-bio"),
        ("MeSH term lookup for diabetes", "pubmed-bio"),
        ("systematic review on GLP-1 agonists", "pubmed-bio"),
        # Non-specific biomedical queries fall back to PubMed (default).
        ("TP53 gene variants in breast cancer", "pubmed-bio"),
        ("side effects of ibuprofen", "pubmed-bio"),
    ],
)
@pytest.mark.asyncio
async def test_bio_keyword_routing(orch, query, expected_server):
    picked = await orch._select_best_mcp_server(list(BIO_MCP_SERVERS), query)
    assert picked == expected_server


# ---------------------------------------------------------------------------
# Multi-server semantic fan-out for the bio umbrella
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query,expected_servers",
    [
        # Cross-domain: literature + trials
        (
            "PubMed systematic review of BRAF clinical trials",
            {"pubmed-bio", "clinicaltrials-bio"},
        ),
        # Cross-domain: compound + trials ("phase III" + "drug structure")
        (
            "phase III trial of a drug structure containing fluorine",
            {"clinicaltrials-bio", "pubchem-bio"},
        ),
        # Cross-domain: all three (rare but valid)
        (
            "PubMed MeSH on CID compound in phase III trial",
            {"pubmed-bio", "clinicaltrials-bio", "pubchem-bio"},
        ),
        # Single-domain remains single
        ("SMILES of aspirin", {"pubchem-bio"}),
        # "melanoma" is biomed-general so PubMed joins ClinicalTrials.
        ("latest NCT trials for melanoma", {"clinicaltrials-bio", "pubmed-bio"}),
        # No explicit keywords → default fallback to pubmed-bio only
        ("aspirin cardiovascular disease", {"pubmed-bio"}),
    ],
)
def test_bio_fanout_semantic(orch, query, expected_servers):
    selected = orch._select_bio_servers(list(BIO_MCP_SERVERS), query)
    assert set(selected) == expected_servers


# ---------------------------------------------------------------------------
# Tool + params mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "server,expected_tool",
    [
        ("pubmed-bio", "pubmed_search_articles"),
        ("clinicaltrials-bio", "clinicaltrials_search_studies"),
        ("pubchem-bio", "pubchem_search_compounds"),
    ],
)
def test_mcp_tool_for_query_bio(orch, server, expected_tool):
    assert orch._get_mcp_tool_for_query(server, "any") == expected_tool


def test_mcp_tool_params_pubmed(orch):
    # pubmed uses `maxResults` (camelCase) per cyanheads schema
    params = orch._get_mcp_tool_params("pubmed-bio", "pubmed_search_articles", "aspirin")
    assert params == {"query": "aspirin", "maxResults": 10}


def test_mcp_tool_params_clinicaltrials(orch):
    # schema accepts a free-text `query`, no explicit max_results
    params = orch._get_mcp_tool_params(
        "clinicaltrials-bio", "clinicaltrials_search_studies", "BRAF V600E"
    )
    assert params == {"query": "BRAF V600E"}


def test_mcp_tool_params_pubchem(orch):
    # pubchem requires searchType + identifierType + identifiers
    params = orch._get_mcp_tool_params(
        "pubchem-bio", "pubchem_search_compounds", "aspirin"
    )
    assert params == {
        "searchType": "identifier",
        "identifierType": "name",
        "identifiers": ["aspirin"],
    }


# ---------------------------------------------------------------------------
# search_bio_tool helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "domain,expected_server,expected_tool",
    [
        ("auto", "pubmed-bio", "pubmed_search_articles"),
        ("literature", "pubmed-bio", "pubmed_search_articles"),
        ("pubmed", "pubmed-bio", "pubmed_search_articles"),
        ("trials", "clinicaltrials-bio", "clinicaltrials_search_studies"),
        ("clinical", "clinicaltrials-bio", "clinicaltrials_search_studies"),
        ("compounds", "pubchem-bio", "pubchem_search_compounds"),
        ("drugs", "pubchem-bio", "pubchem_search_compounds"),
        # Unknown domain falls back to literature
        ("nonsense", "pubmed-bio", "pubmed_search_articles"),
    ],
)
def test_search_bio_tool_dispatch(domain, expected_server, expected_tool):
    record = search_bio_tool("test query", domain=domain, max_results=5)
    assert record["mcp_server"] == expected_server
    assert record["tool_name"] == expected_tool
    params = record["params"]
    # Param shape depends on server.
    if expected_server == "pubmed-bio":
        assert params == {"query": "test query", "maxResults": 5}
    elif expected_server == "clinicaltrials-bio":
        assert params == {"query": "test query"}
    else:  # pubchem-bio
        assert params == {
            "searchType": "identifier",
            "identifierType": "name",
            "identifiers": ["test query"],
        }


def test_search_bio_tool_caps_max_results():
    # Only PubMed carries a max_results through; cap applies there.
    record = search_bio_tool("x", domain="literature", max_results=999)
    assert record["params"]["maxResults"] == 50


# ---------------------------------------------------------------------------
# Semantic intent weights (compute_intent_weights)
# ---------------------------------------------------------------------------

def _approx_sum_one(weights):
    return abs(sum(weights.values()) - 1.0) < 0.01


def test_weights_sum_to_one_and_respect_web_floor():
    w = compute_intent_weights("RFdiffusion de novo binder protein design")
    assert _approx_sum_one(w)
    # Web floor 10% enforced even when query is purely bio.
    assert w["web"] >= 0.1 - 0.005
    # Bio should dominate.
    assert w["bio"] > w["ai"]
    assert w["bio"] > w["web"]


def test_weights_ai_dominant_query():
    w = compute_intent_weights(
        "LLM chain-of-thought faithfulness transformer reinforcement learning"
    )
    assert _approx_sum_one(w)
    assert w["ai"] > w["bio"]
    assert w["ai"] > w["web"]
    # No zero-sum: bio gets the 5% floor.
    assert w["bio"] >= 0.05 - 0.005
    assert w["web"] >= 0.1 - 0.005


def test_weights_mixed_ai_and_bio():
    # User-requested example: 60/30/10 split for AI+Bio mixed query.
    w = compute_intent_weights(
        "comparison of LLMs and protein transformer models for drug discovery"
    )
    assert _approx_sum_one(w)
    # AI should lead, bio second, web floor at 10%.
    assert w["ai"] > w["bio"] > 0.1
    assert w["web"] >= 0.1 - 0.005


def test_weights_allowed_subset_only_bio():
    # User ticked only Bio — no floor added for groups not in allowed set.
    w = compute_intent_weights("PubMed systematic review", allowed_groups={"bio"})
    assert w == {"bio": 1.0}


def test_weights_allowed_subset_ai_and_web_only():
    w = compute_intent_weights(
        "latest transformer scaling law", allowed_groups={"ai", "web"}
    )
    assert _approx_sum_one(w)
    assert "bio" not in w  # bio was not allowed
    assert w["web"] >= 0.1 - 0.005


def test_weights_zero_keyword_query_splits_evenly():
    w = compute_intent_weights("hello")  # no keywords match
    assert _approx_sum_one(w)
    # Should be roughly 1/3 each after floors (web floor may nudge).
    assert all(v > 0 for v in w.values())
