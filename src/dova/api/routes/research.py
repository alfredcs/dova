"""
Research Endpoints for DOVA API.
"""

import json
from typing import Any

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from dova.api.schemas.research import (
    ResearchRequest,
    ResearchResponse,
    SearchRequest,
    SearchResponse,
)
from dova.api.middleware.auth import get_current_user, User

router = APIRouter()
logger = structlog.get_logger(__name__)


def _extract_agent_data(result_data: Any) -> dict:
    """Map direct research_agent output (ResearchFindings) to flat dict."""
    out: dict[str, Any] = {
        "response": "",
        "papers": [],
        "repositories": [],
        "models": [],
        "datasets": [],
        "web_results": [],
        "images": [],
        "mcp_results": [],
        "deliberation": {},
    }
    if result_data is None:
        return out
    for field, key in [
        ("papers", "papers"),
        ("repositories", "repositories"),
        ("models", "models"),
        ("datasets", "datasets"),
        ("web_results", "web_results"),
    ]:
        items = getattr(result_data, field, None)
        if items:
            out[key] = [
                {"title": i.title, "url": i.url, "description": i.description, "metadata": i.metadata}
                for i in items
            ]
    return out


def _derive_confidence(deliberation: dict, has_results: bool) -> float:
    """Derive confidence from deliberation metadata and result presence."""
    action = deliberation.get("action", "")
    if action == "use_tools":
        return 0.8 if has_results else 0.4
    if action == "respond_directly":
        return 0.7
    return 0.5


@router.post("/research", response_model=ResearchResponse)
async def execute_research(
    request: Request,
    body: ResearchRequest,
    current_user: User = Depends(get_current_user),
) -> ResearchResponse:
    """
    Execute a comprehensive research query.

    Delegates to ThinkingOrchestrator for deliberation-first orchestration,
    then maps results back to the ResearchResponse contract.
    """
    logger.info(
        "research_request",
        user_id=current_user.id,
        query=body.query,
        sources=body.sources,
    )

    try:
        from dova.agents.base import AgentTask
        from dova.agents.thinking_orchestrator import ThinkingOrchestrator

        orchestrator = getattr(request.app.state, "orchestrator", None)
        research_agent = getattr(request.app.state, "research_agent", None)

        if orchestrator is None and research_agent is None:
            raise HTTPException(
                status_code=503,
                detail="Research service not available",
            )

        task = AgentTask(
            type="research",
            params={
                "query": body.query,
                "sources": body.sources,
                "max_results": body.max_results,
            },
            user_id=current_user.id,
        )

        # Route to orchestrator (default) or direct agent
        use_orchestrator = body.orchestrator == "thinking" and orchestrator is not None
        if use_orchestrator:
            result = await orchestrator.execute(task)
        elif research_agent is not None:
            result = await research_agent.execute(task)
        else:
            raise HTTPException(
                status_code=503,
                detail="Research service not available",
            )

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)

        # Extract flat research arrays from orchestrator output
        if use_orchestrator:
            data = ThinkingOrchestrator.extract_research_data(result.data or {})
        else:
            data = _extract_agent_data(result.data)

        deliberation = data.get("deliberation", {})
        papers = data.get("papers", [])
        repositories = data.get("repositories", [])
        models = data.get("models", [])
        datasets = data.get("datasets", [])
        web_results = data.get("web_results", [])
        images = data.get("images", [])
        answer = data.get("response", "")

        # Build summary
        found_parts = []
        if papers:
            found_parts.append(f"{len(papers)} papers")
        if repositories:
            found_parts.append(f"{len(repositories)} repositories")
        if models:
            found_parts.append(f"{len(models)} models")
        if datasets:
            found_parts.append(f"{len(datasets)} datasets")
        if web_results:
            found_parts.append(f"{len(web_results)} web results")
        summary = f"Found {', '.join(found_parts)}" if found_parts else "No results found"

        has_results = bool(papers or repositories or models or datasets or web_results)
        confidence = _derive_confidence(deliberation, has_results)

        # Convert image dicts to ImageResult schema
        from dova.api.schemas.research import ImageResult
        image_results = [
            ImageResult(
                url=img.get("url", ""),
                prompt=img.get("prompt", ""),
                resolution=img.get("resolution", "1024x1024"),
                seed=img.get("seed", 0),
            )
            for img in images
            if isinstance(img, dict)
        ]

        # Derive insights from deliberation knowledge_gaps
        knowledge_gaps = deliberation.get("knowledge_gaps", [])
        insights = [f"Knowledge gap identified: {gap}" for gap in knowledge_gaps[:5]] if knowledge_gaps else []

        # Derive recommendations from tools_used
        tools_used = deliberation.get("tools_used", [])
        recommendations = []
        if not has_results and tools_used:
            recommendations.append("Try broadening your query or adding more sources.")
        if knowledge_gaps:
            recommendations.append("Consider a follow-up query to fill the identified knowledge gaps.")

        return ResearchResponse(
            query=body.query,
            status="completed",
            answer=answer,
            summary=summary,
            papers=papers,
            repositories=repositories,
            models=models,
            datasets=datasets,
            web_results=web_results,
            images=image_results,
            insights=insights,
            recommendations=recommendations,
            confidence=confidence,
            refinement_attempts=0,
            reasoning_trace=[],
            debate={},
            metadata={
                "execution_time_ms": result.execution_time_ms,
                "sources_searched": body.sources,
                "orchestrator": body.orchestrator,
                "deliberation_action": deliberation.get("action", ""),
                "tools_used": tools_used,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("research_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/research/upload", response_model=ResearchResponse)
async def execute_research_with_files(
    request: Request,
    query: str = Form(...),
    sources: str = Form(default='["arxiv","github","huggingface"]'),
    max_results: int = Form(default=20),
    orchestrator: str = Form(default="thinking"),
    files: list[UploadFile] = File(default=[]),
    current_user: User = Depends(get_current_user),
) -> ResearchResponse:
    """
    Execute a research query with optional file attachments.

    Accepts multipart form data with files (.txt, .pdf, .png).
    File contents are extracted and appended to the query.
    """
    from dova.services.file_processor import MAX_FILES, process_uploaded_file

    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_FILES} files allowed",
        )

    # Parse sources JSON string
    try:
        parsed_sources = json.loads(sources)
    except (json.JSONDecodeError, TypeError):
        parsed_sources = ["arxiv", "github", "huggingface"]

    # Process attached files and combine with query
    combined_query = query
    if files:
        file_parts = []
        for f in files:
            content = await process_uploaded_file(f)
            file_parts.append(f"[File: {f.filename}]\n{content}")
        attached = "\n\n".join(file_parts)
        combined_query = f"{query}\n\n--- Attached Files ---\n\n{attached}"

    # Build body bypassing max_length validation (combined query includes file content)
    body = ResearchRequest.model_construct(
        query=combined_query,
        sources=parsed_sources,
        max_results=max_results,
        orchestrator=orchestrator,
    )
    return await execute_research(request, body, current_user)


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
