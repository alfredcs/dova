"""
Dova MCP Server for Claude Code.

Exposes Dova's multi-agent research capabilities as MCP tools:
- dova_research: Full orchestrated research across all sources
- dova_search: Single-source search (arxiv, github, huggingface, web)
- dova_debate: Bull vs Bear structured debate
- dova_validate: Code quality and security analysis
- dova_web_search: Multi-provider web search
"""

import json
from dataclasses import asdict, dataclass
from typing import Any

import structlog
from mcp.server.fastmcp import FastMCP

from dova.agents.base import AgentTask

logger = structlog.get_logger(__name__)

mcp = FastMCP(
    "dova",
    instructions="Multi-agent AI/ML research platform — search ArXiv, GitHub, HuggingFace, and the web",
)

# Lazy-initialized services
_services: dict[str, Any] = {}


async def _get_services() -> dict[str, Any]:
    """Lazy-init Dova services. Handles missing credentials gracefully."""
    if _services:
        return _services

    from dotenv import load_dotenv

    load_dotenv()

    # LLM Router
    try:
        from dova.config.providers import create_llm_router_from_settings

        _services["llm_router"] = create_llm_router_from_settings()
    except Exception as e:
        logger.warning("llm_router_init_failed", error=str(e))
        _services["llm_router"] = None

    # Web search
    try:
        from dova.services.web_search import create_parallel_search_service

        _services["web_search"] = create_parallel_search_service()
    except Exception as e:
        logger.warning("web_search_init_failed", error=str(e))
        _services["web_search"] = None

    return _services


def _serialize(obj: Any) -> Any:
    """Serialize dataclasses and other objects to JSON-safe dicts."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    return obj


@mcp.tool()
async def dova_research(
    query: str,
    sources: str = "all",
    max_results: int = 10,
) -> str:
    """Research a topic across ArXiv, GitHub, HuggingFace, and the web.

    Args:
        query: Research query (e.g. "transformer architectures for NLP")
        sources: Comma-separated sources: arxiv,github,huggingface,web or "all"
        max_results: Maximum results per source
    """
    services = await _get_services()
    llm_router = services.get("llm_router")
    if not llm_router:
        return json.dumps({"error": "LLM router not available. Check provider credentials."})

    try:
        from dova.agents.research import ResearchAgent

        agent = ResearchAgent(llm_router=llm_router)
        task = AgentTask(
            type="research",
            params={
                "query": query,
                "source": sources.strip(),
                "max_results": max_results,
            },
        )
        result = await agent.execute(task)
        return json.dumps(_serialize(result.data) if result.success else {"error": result.error})
    except Exception as e:
        logger.exception("dova_research_error")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def dova_search(
    query: str,
    source: str = "arxiv",
    max_results: int = 10,
) -> str:
    """Search a single source for research results.

    Args:
        query: Search query
        source: One of: arxiv, github, huggingface, web
        max_results: Maximum results to return
    """
    valid_sources = {"arxiv", "github", "huggingface", "web"}
    if source not in valid_sources:
        return json.dumps({"error": f"Invalid source '{source}'. Choose from: {', '.join(sorted(valid_sources))}"})

    services = await _get_services()
    llm_router = services.get("llm_router")
    if not llm_router:
        return json.dumps({"error": "LLM router not available. Check provider credentials."})

    try:
        from dova.agents.research import ResearchAgent

        agent = ResearchAgent(llm_router=llm_router)
        task = AgentTask(
            type="research",
            params={
                "query": query,
                "source": source,
                "max_results": max_results,
            },
        )
        result = await agent.execute(task)
        return json.dumps(_serialize(result.data) if result.success else {"error": result.error})
    except Exception as e:
        logger.exception("dova_search_error")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def dova_debate(
    topic: str,
    num_rounds: int = 2,
) -> str:
    """Run a structured Bull vs Bear debate on a topic.

    Args:
        topic: Topic to debate (e.g. "Is RLHF the best alignment approach?")
        num_rounds: Number of debate rounds (1-5)
    """
    num_rounds = max(1, min(5, num_rounds))

    services = await _get_services()
    llm_router = services.get("llm_router")
    if not llm_router:
        return json.dumps({"error": "LLM router not available. Check provider credentials."})

    try:
        from dova.agents.debate import DebateAgent

        agent = DebateAgent(llm_router=llm_router, num_rounds=num_rounds)
        task = AgentTask(
            type="debate",
            params={"topic": topic},
        )
        result = await agent.execute(task)
        return json.dumps(_serialize(result.data) if result.success else {"error": result.error})
    except Exception as e:
        logger.exception("dova_debate_error")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def dova_validate(
    code: str,
    language: str = "python",
) -> str:
    """Validate code for quality, security, and correctness.

    Args:
        code: Source code to validate
        language: Programming language (python, javascript, typescript, etc.)
    """
    services = await _get_services()
    llm_router = services.get("llm_router")
    if not llm_router:
        return json.dumps({"error": "LLM router not available. Check provider credentials."})

    try:
        from dova.agents.validation import ValidationAgent

        agent = ValidationAgent(llm_router=llm_router)
        task = AgentTask(
            type="analyze",
            params={
                "code": code,
                "language": language,
            },
        )
        result = await agent.execute(task)
        return json.dumps(_serialize(result.data) if result.success else {"error": result.error})
    except Exception as e:
        logger.exception("dova_validate_error")
        return json.dumps({"error": str(e)})


@mcp.tool()
async def dova_web_search(
    query: str,
    max_results: int = 10,
) -> str:
    """Search the web using multiple providers (Brave, Perplexity, Tavily, DuckDuckGo).

    Args:
        query: Search query
        max_results: Maximum results to return
    """
    services = await _get_services()
    web_search = services.get("web_search")

    if not web_search:
        return json.dumps({"error": "Web search service not available."})

    try:
        response = await web_search.search(query=query, max_results=max_results)
        return json.dumps(_serialize(response))
    except Exception as e:
        logger.exception("dova_web_search_error")
        return json.dumps({"error": str(e)})
