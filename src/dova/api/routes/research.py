"""
Research Endpoints for DOVA API.
"""

from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from dova.api.schemas.research import (
    ResearchRequest,
    ResearchResponse,
    SearchRequest,
    SearchResponse,
)
from dova.api.middleware.auth import get_current_user, User

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post("/research", response_model=ResearchResponse)
async def execute_research(
    request: Request,
    body: ResearchRequest,
    current_user: User = Depends(get_current_user),
) -> ResearchResponse:
    """
    Execute a comprehensive research query.

    This endpoint orchestrates searches across ArXiv, GitHub, and HuggingFace,
    synthesizes the results, and returns personalized insights.

    Args:
        body: Research request with query and options
        current_user: Authenticated user

    Returns:
        Synthesized research results
    """
    logger.info(
        "research_request",
        user_id=current_user.id,
        query=body.query,
        sources=body.sources,
    )

    try:
        # Get orchestrator from app state (would be initialized in lifespan)
        orchestrator = getattr(request.app.state, "orchestrator", None)

        if orchestrator is None:
            # Return mock response for now
            return ResearchResponse(
                query=body.query,
                status="completed",
                summary="Research query received. Orchestrator not initialized.",
                papers=[],
                repositories=[],
                models=[],
                insights=[],
                recommendations=[],
            )

        # Execute research through orchestrator
        from dova.agents.base import AgentTask

        task = AgentTask(
            type="research",
            params={
                "query": body.query,
                "sources": body.sources,
                "max_results": body.max_results,
            },
            user_id=current_user.id,
        )

        result = await orchestrator.execute(task)

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)

        data = result.data or {}
        return ResearchResponse(
            query=body.query,
            status="completed",
            summary=data.get("summary", ""),
            papers=data.get("papers", []),
            repositories=data.get("repositories", []),
            models=data.get("models", []),
            insights=data.get("insights", []),
            recommendations=data.get("recommendations", []),
            metadata={
                "execution_time_ms": result.execution_time_ms,
                "sources_searched": body.sources,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("research_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/{source}", response_model=SearchResponse)
async def search_source(
    request: Request,
    source: str,
    body: SearchRequest,
    current_user: User = Depends(get_current_user),
) -> SearchResponse:
    """
    Search a specific source (arxiv, github, huggingface).

    Args:
        source: Source to search (arxiv, github, huggingface)
        body: Search request with query and filters
        current_user: Authenticated user

    Returns:
        Search results from the specified source
    """
    valid_sources = ["arxiv", "github", "huggingface"]
    if source not in valid_sources:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source. Must be one of: {valid_sources}",
        )

    logger.info(
        "search_request",
        user_id=current_user.id,
        source=source,
        query=body.query,
    )

    try:
        # Get research agent from app state
        research_agent = getattr(request.app.state, "research_agent", None)

        if research_agent is None:
            # Return mock response
            return SearchResponse(
                source=source,
                query=body.query,
                results=[],
                total_count=0,
                metadata={"status": "agent_not_initialized"},
            )

        from dova.agents.base import AgentTask

        task = AgentTask(
            type="search",
            params={
                "source": source,
                "query": body.query,
                "max_results": body.max_results,
                "filters": body.filters,
            },
            user_id=current_user.id,
        )

        result = await research_agent.execute(task)

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)

        data = result.data
        results = []

        # Extract results based on source
        if hasattr(data, "papers"):
            results.extend(
                {
                    "title": p.title,
                    "url": p.url,
                    "description": p.description,
                    "metadata": p.metadata,
                }
                for p in data.papers
            )
        if hasattr(data, "repositories"):
            results.extend(
                {
                    "title": r.title,
                    "url": r.url,
                    "description": r.description,
                    "metadata": r.metadata,
                }
                for r in data.repositories
            )
        if hasattr(data, "models"):
            results.extend(
                {
                    "title": m.title,
                    "url": m.url,
                    "description": m.description,
                    "metadata": m.metadata,
                }
                for m in data.models
            )

        return SearchResponse(
            source=source,
            query=body.query,
            results=results,
            total_count=len(results),
            metadata={
                "execution_time_ms": result.execution_time_ms,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("search_error", source=source, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/debate")
async def run_debate(
    request: Request,
    topic: str,
    context: dict[str, Any] | None = None,
    num_rounds: int = 2,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Run a Bull vs Bear debate on a topic.

    Args:
        topic: Topic to debate
        context: Optional context for the debate
        num_rounds: Number of debate rounds (default 2)
        current_user: Authenticated user

    Returns:
        Debate results with balanced conclusion
    """
    logger.info(
        "debate_request",
        user_id=current_user.id,
        topic=topic,
        num_rounds=num_rounds,
    )

    try:
        debate_agent = getattr(request.app.state, "debate_agent", None)

        if debate_agent is None:
            return {
                "status": "error",
                "message": "Debate agent not initialized",
            }

        from dova.agents.base import AgentTask

        task = AgentTask(
            type="debate",
            params={
                "topic": topic,
                "context": context or {},
            },
            user_id=current_user.id,
        )

        result = await debate_agent.execute(task)

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)

        return result.data

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("debate_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
