"""
Dova MCP Server for Claude Code.

Exposes Dova's multi-agent research capabilities as MCP tools:
- dova_research: Full orchestrated research across all sources
- dova_search: Single-source search (arxiv, github, huggingface, web)
- dova_debate: Bull vs Bear structured debate
- dova_validate: Code quality and security analysis
- dova_web_search: Multi-provider web search
"""

import asyncio
import json
from dataclasses import asdict, dataclass
from typing import Any

import structlog
from mcp.server.fastmcp import FastMCP

from dova.agents.base import AgentTask
from dova.utils.concurrency import tracked

logger = structlog.get_logger(__name__)

mcp = FastMCP(
    "dova",
    instructions="Multi-agent AI/ML research platform — search ArXiv, GitHub, HuggingFace, and the web",
)

# Lazy-initialized services. `_services_lock` guards cold-start so concurrent
# callers don't each run the 140-line init block (which spawns subprocesses,
# clones git repos, and builds LLM clients). Once `_services` is populated
# the lock is uncontended — the fast path returns immediately without it.
_services: dict[str, Any] = {}
_services_lock: asyncio.Lock | None = None


async def _get_services() -> dict[str, Any]:
    """Lazy-init Dova services. Handles missing credentials gracefully.

    Mirrors the wiring in ``dova.api.main.lifespan`` so the MCP server and the
    FastAPI server share an identical deliberation-first orchestration path.

    Concurrency: the init block is serialized via ``_services_lock`` so
    parallel MCP requests at cold start don't each spawn subprocesses,
    clone repos, or instantiate LLM clients independently. Hot-path returns
    without acquiring the lock.
    """
    if _services:
        return _services

    global _services_lock
    if _services_lock is None:
        _services_lock = asyncio.Lock()

    async with _services_lock:
        # Double-check after acquiring the lock — another coroutine may
        # have finished init while we were waiting.
        if _services:
            return _services
        await _init_services()
    return _services


async def _init_services() -> None:
    """Populate ``_services`` in-place. Must be called with ``_services_lock`` held."""
    from dotenv import load_dotenv

    load_dotenv()

    # Enlarge the event loop's default ThreadPoolExecutor before we start
    # routing blocking boto3 calls through it. Idempotent.
    from dova.utils.concurrency import configure_default_executor, start_saturation_logger

    configure_default_executor()
    # Log executor + in-flight-request saturation every 5s while requests
    # are active. Silent when idle. Gives us real data on whether 64
    # workers is enough for observed traffic.
    start_saturation_logger()

    from dova.config.settings import get_settings

    settings = get_settings()

    # Enhanced memory (in-memory cache for local/dev; orchestrator uses this
    # to persist ConversationContext across calls within a session).
    try:
        from dova.services.memory_enhanced import EnhancedMemoryService
        from dova.utils.cache import InMemoryCache, RedisCache

        if settings.is_production:
            memory_cache = RedisCache(url=settings.redis.url, prefix="dova:memory:")
        else:
            memory_cache = InMemoryCache()
        _services["enhanced_memory_service"] = EnhancedMemoryService(
            cache=memory_cache,
            llm_router=None,  # wired below after LLM router init
            mmr_lambda=settings.memory_enhanced.mmr_lambda,
            embedding_cache_ttl=settings.memory_enhanced.embedding_cache_ttl,
        )
    except Exception as e:
        logger.warning("enhanced_memory_init_failed", error=str(e))
        _services["enhanced_memory_service"] = None

    # AgentCore memory (only when AWS is configured)
    try:
        if settings.aws.agentcore_memory_enabled and settings.aws.agentcore_agent_id:
            from dova.services.memory import AgentCoreMemoryService

            _services["memory_service"] = AgentCoreMemoryService(
                agent_id=settings.aws.agentcore_agent_id,
                agent_alias_id=settings.aws.agentcore_agent_alias_id or "",
                region=settings.aws.region,
            )
        else:
            _services["memory_service"] = None
    except Exception as e:
        logger.warning("memory_service_init_failed", error=str(e))
        _services["memory_service"] = None

    # Source registry
    try:
        from dova.services.sources import SourceRegistry

        _services["source_registry"] = SourceRegistry(
            memory_service=_services.get("memory_service"),
        )
    except Exception as e:
        logger.warning("source_registry_init_failed", error=str(e))
        _services["source_registry"] = None

    # LLM Router
    try:
        from dova.config.providers import create_llm_router_from_settings

        _services["llm_router"] = create_llm_router_from_settings()
    except Exception as e:
        logger.warning("llm_router_init_failed", error=str(e))
        _services["llm_router"] = None

    # Wire LLM router into enhanced memory for embedding support
    if _services.get("enhanced_memory_service") and _services.get("llm_router"):
        _services["enhanced_memory_service"].llm_router = _services["llm_router"]

    # MCP client (for arxiv/github/huggingface/bio tool access)
    try:
        from dova.tools.mcp_registry import MCPManager

        _services["mcp_manager"] = MCPManager()
        _services["mcp_client"] = _services["mcp_manager"].get_client()
    except Exception as e:
        logger.warning("mcp_client_init_failed", error=str(e))
        _services["mcp_client"] = None

    # Setup MCP server repos (clone/update arxiv-mcp-server etc.) — matches
    # dova.api.main.lifespan so the arxiv STDIO server is available.
    try:
        from dova.services.mcp_repo_manager import setup_mcp_repos

        mcp_repo_results = await setup_mcp_repos()
        logger.info("mcp_repos_setup", results=mcp_repo_results)
    except Exception as e:
        logger.warning("mcp_repos_setup_failed", error=str(e))

    # Web search
    try:
        from dova.services.web_search import create_parallel_search_service

        _services["web_search"] = create_parallel_search_service()
    except Exception as e:
        logger.warning("web_search_init_failed", error=str(e))
        _services["web_search"] = None

    # ThinkingOrchestrator — same path used by the web UI's /api/research
    llm_router = _services.get("llm_router")
    if llm_router is not None:
        try:
            from dova.agents.debate import DebateAgent
            from dova.agents.research import ResearchAgent
            from dova.agents.thinking_orchestrator import ThinkingOrchestrator

            research_agent = ResearchAgent(
                llm_router=llm_router,
                mcp_client=_services.get("mcp_client"),
                memory_service=_services.get("memory_service"),
                source_registry=_services.get("source_registry"),
                tavily_api_key=settings.mcp.tavily_api_key,
                enhanced_memory_service=_services.get("enhanced_memory_service"),
            )
            debate_agent = DebateAgent(
                llm_router=llm_router,
                mcp_client=_services.get("mcp_client"),
                num_rounds=2,
            )
            orchestrator = ThinkingOrchestrator(
                llm_router=llm_router,
                mcp_client=_services.get("mcp_client"),
                memory_service=_services.get("enhanced_memory_service"),
                web_search_service=_services.get("web_search"),
            )
            orchestrator.register_agent("research", research_agent)
            orchestrator.register_agent("debate", debate_agent)
            _services["orchestrator"] = orchestrator
            _services["research_agent"] = research_agent
            _services["debate_agent"] = debate_agent
        except Exception as e:
            logger.warning("orchestrator_init_failed", error=str(e))
            _services["orchestrator"] = None


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
@tracked("dova_research")
async def dova_research(
    query: str,
    sources: str = "all",
    max_results: int = 20,
) -> str:
    """Research a topic across ArXiv, GitHub, HuggingFace, the web, and biomedical sources.

    Returns the same synthesized payload as DOVA's web UI: a long-form
    `answer` (with LaTeX formulas and IEEE-style `algorithm` blocks where
    relevant, plus novel ideas), flat source arrays, and deliberation metadata.

    Args:
        query: Research query (e.g. "transformer architectures for NLP")
        sources: Comma-separated list. Accepts top-level groups (ai, web, bio)
            and/or concrete sources (arxiv, github, huggingface, web, bio).
            The "ai" group expands to arxiv+github+huggingface; "bio" fans out
            to PubMed, ClinicalTrials.gov, and PubChem. Use "all" for everything.
        max_results: Maximum results per source
    """
    services = await _get_services()
    orchestrator = services.get("orchestrator")
    if orchestrator is None:
        return json.dumps({"error": "Orchestrator not available. Check provider credentials."})

    try:
        from dova.agents.thinking_orchestrator import ThinkingOrchestrator

        # Mirror the web UI's SOURCE_GROUP_MAP (api/static/index.html).
        group_map = {
            "ai": ["arxiv", "github", "huggingface"],
            "web": ["web"],
            "bio": ["bio"],
        }
        src = sources.strip()
        if src in ("", "all"):
            raw_tokens = ["ai", "web", "bio"]
        else:
            raw_tokens = [s.strip() for s in src.split(",") if s.strip()]

        seen: set[str] = set()
        source_list: list[str] = []
        for tok in raw_tokens:
            for s in group_map.get(tok, [tok]):
                if s not in seen:
                    seen.add(s)
                    source_list.append(s)

        task = AgentTask(
            type="research",
            params={
                "query": query,
                "sources": source_list,
                "max_results": max_results,
            },
        )
        result = await orchestrator.execute(task)
        if not result.success:
            return json.dumps({"error": result.error})

        data = ThinkingOrchestrator.extract_research_data(result.data or {})
        return json.dumps(_serialize({
            "query": query,
            "answer": data.get("response", ""),
            "top_papers": data.get("top_papers", []),
            "papers": data.get("papers", []),
            "pubmed_papers": data.get("pubmed_papers", []),
            "cross_domain_bridges": data.get("cross_domain_bridges", []),
            "drug_story": data.get("drug_story"),
            "repositories": data.get("repositories", []),
            "models": data.get("models", []),
            "datasets": data.get("datasets", []),
            "web_results": data.get("web_results", []),
            "images": data.get("images", []),
            "deliberation": data.get("deliberation", {}),
        }))
    except Exception as e:
        logger.exception("dova_research_error")
        return json.dumps({"error": str(e)})


@mcp.tool()
@tracked("dova_search")
async def dova_search(
    query: str,
    source: str = "arxiv",
    max_results: int = 20,
) -> str:
    """Search a single source for research results.

    Args:
        query: Search query
        source: One of: arxiv, github, huggingface, web, bio
        max_results: Maximum results to return
    """
    valid_sources = {"arxiv", "github", "huggingface", "web", "bio"}
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
@tracked("dova_debate")
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
@tracked("dova_validate")
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
@tracked("dova_web_search")
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
        response = await web_search.search_parallel(query=query, max_results=max_results)
        return json.dumps(_serialize(response))
    except Exception as e:
        logger.exception("dova_web_search_error")
        return json.dumps({"error": str(e)})
