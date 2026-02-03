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


async def _critique_answer(
    llm_router: Any,
    query: str,
    answer: str,
    sources_found: dict[str, int],
) -> dict[str, Any]:
    """
    Critique the synthesized answer and provide confidence score.

    Returns:
        dict with:
        - confidence: 0.0-1.0 score
        - is_sufficient: bool
        - missing_info: list of missing information
        - refined_query: suggested refined query if confidence is low
    """
    critique_prompt = f"""You are a research quality evaluator. Analyze the following answer to determine if it adequately addresses the user's question.

**User Question:** {query}

**Generated Answer:**
{answer}

**Sources Found:** {sources_found}

**Evaluation Criteria:**
1. Does the answer directly address the question?
2. Does it provide specific facts, names, dates, or numbers?
3. Is the information comprehensive or superficial?
4. Does the answer acknowledge uncertainty or gaps?

**Respond in this exact JSON format:**
{{
    "confidence": <0.0 to 1.0>,
    "is_sufficient": <true or false>,
    "assessment": "<brief explanation of confidence score>",
    "missing_info": ["<info 1>", "<info 2>"],
    "refined_query": "<better search query if confidence < 0.7, else empty string>"
}}

**JSON Response:**"""

    try:
        from dova.config.providers import LLMRequest, TaskType
        import json

        request = LLMRequest(
            task_type=TaskType.CHAT,
            messages=[{"role": "user", "content": critique_prompt}],
            max_tokens=4000,
            temperature=0.2,
        )
        response = await llm_router.complete(request)
        content = response.content.strip()

        # Parse JSON from response
        # Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        critique = json.loads(content)
        return {
            "confidence": float(critique.get("confidence", 0.5)),
            "is_sufficient": critique.get("is_sufficient", False),
            "assessment": critique.get("assessment", ""),
            "missing_info": critique.get("missing_info", []),
            "refined_query": critique.get("refined_query", ""),
        }
    except Exception as e:
        logger.warning("critique_failed", error=str(e))
        # Default to moderate confidence if critique fails
        return {
            "confidence": 0.5,
            "is_sufficient": True,
            "assessment": "Critique unavailable",
            "missing_info": [],
            "refined_query": "",
        }


async def _retrieve_relevant_memory(
    memory_service: Any,
    query: str,
    user_id: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Retrieve relevant past research from memory using semantic search.

    Returns:
        List of relevant memory entries with query, answer_summary, confidence
    """
    if not memory_service:
        return []

    try:
        from dova.services.memory_enhanced import MemoryType

        # Search for semantically similar past research
        results = await memory_service.search_semantic(
            query=query,
            user_id=user_id,
            top_k=top_k,
            use_mmr=True,  # Use MMR for diversity
            memory_types=[MemoryType.LONG_TERM, MemoryType.SHORT_TERM],
        )

        relevant = []
        for result in results:
            content = result.entry.content
            if content.get("type") == "research_result":
                relevant.append({
                    "query": content.get("query", ""),
                    "answer_summary": content.get("answer_summary", ""),
                    "confidence": content.get("confidence", 0),
                    "similarity_score": result.score,
                })

        if relevant:
            logger.info(
                "memory_retrieved",
                query=query[:50],
                matches_found=len(relevant),
                top_score=relevant[0]["similarity_score"] if relevant else 0,
            )

        return relevant

    except Exception as e:
        logger.warning("memory_retrieval_failed", error=str(e))
        return []


async def _store_research_memory(
    memory_service: Any,
    query: str,
    answer: str,
    confidence: float,
    user_id: str,
) -> None:
    """Store research results in long-term memory for future reference."""
    if not memory_service:
        return

    try:
        from dova.services.memory_enhanced import MemoryType

        # Store high-confidence answers in long-term memory
        if confidence >= 0.7:
            await memory_service.store(
                memory_type=MemoryType.LONG_TERM,
                content={
                    "type": "research_result",
                    "query": query,
                    "answer_summary": answer[:500],
                    "confidence": confidence,
                },
                importance=confidence,
                user_id=user_id,
                tags=["research", "verified"],
            )
            logger.info("research_stored_long_term", query=query[:50], confidence=confidence)
        else:
            # Store low-confidence in short-term for refinement
            await memory_service.store(
                memory_type=MemoryType.SHORT_TERM,
                content={
                    "type": "research_attempt",
                    "query": query,
                    "confidence": confidence,
                },
                importance=0.3,
                user_id=user_id,
                tags=["research", "needs_refinement"],
            )
    except Exception as e:
        logger.warning("memory_store_failed", error=str(e))


async def _synthesize_answer(
    llm_router: Any,
    query: str,
    papers: list[dict],
    repositories: list[dict],
    models: list[dict],
    web_results: list[dict],
    memory_context: list[dict] | None = None,
) -> str:
    """
    Synthesize a direct answer from research findings using LLM.

    Args:
        llm_router: LLM router for generating response
        query: Original user query
        papers: Top ArXiv papers found
        repositories: Top GitHub repositories found
        models: Top HuggingFace models found
        web_results: Top web search results
        memory_context: Relevant past research from memory

    Returns:
        Synthesized answer addressing the query
    """
    # Build context from findings
    context_parts = []

    # Add relevant past research from memory
    if memory_context:
        memory_text = "\n".join([
            f"- **Previous research** (confidence: {m.get('confidence', 0):.0%}): {m.get('answer_summary', '')[:200]}..."
            for m in memory_context[:3]
        ])
        context_parts.append(f"**Relevant Past Research:**\n{memory_text}")

    if papers:
        papers_text = "\n".join([
            f"- **{p['title']}**: {p.get('description', '')[:200]}..."
            for p in papers[:5]
        ])
        context_parts.append(f"**Research Papers:**\n{papers_text}")

    if repositories:
        repos_text = "\n".join([
            f"- **{r['title']}** ({r.get('metadata', {}).get('stars', 0):,} stars): {r.get('description', '')[:150]}"
            for r in sorted(repositories, key=lambda x: x.get('metadata', {}).get('stars', 0), reverse=True)[:5]
        ])
        context_parts.append(f"**GitHub Repositories:**\n{repos_text}")

    if models:
        models_text = "\n".join([
            f"- **{m['title']}**: {m.get('description', '')[:150]}"
            for m in models[:3]
        ])
        context_parts.append(f"**AI Models:**\n{models_text}")

    if web_results:
        web_text = "\n".join([
            f"- **{w['title']}**: {w.get('description', '')[:150]}"
            for w in web_results[:5]
        ])
        context_parts.append(f"**Web Sources:**\n{web_text}")

    context = "\n\n".join(context_parts)

    # Create synthesis prompt
    synthesis_prompt = f"""You are a research assistant. Based on the following research findings, provide a direct, comprehensive answer to the user's question.

**User Question:** {query}

**Research Findings:**
{context}

**Instructions:**
1. Provide a direct answer to the question (2-4 paragraphs)
2. Highlight the most relevant findings
3. If asking about "most starred" or "popular" items, list the top results with their metrics
4. Include specific names, numbers, and facts from the findings
5. Be concise but thorough

**Answer:**"""

    try:
        from dova.config.providers import LLMRequest, TaskType

        request = LLMRequest(
            task_type=TaskType.CHAT,
            messages=[{"role": "user", "content": synthesis_prompt}],
            max_tokens=8000,
            temperature=0.3,
        )
        response = await llm_router.complete(request)
        return response.content.strip()
    except Exception as e:
        logger.warning("synthesis_failed", error=str(e))
        return ""


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
        from dova.agents.base import AgentTask

        # Try orchestrator first, then fall back to research_agent
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
                "source": "all" if not body.sources else body.sources[0] if len(body.sources) == 1 else "all",
                "sources": body.sources,
                "max_results": body.max_results,
                "reasoning_mode": body.reasoning_mode,
                "max_reasoning_iterations": body.max_reasoning_iterations,
            },
            user_id=current_user.id,
        )

        # Use orchestrator if available, otherwise use research_agent directly
        if orchestrator:
            result = await orchestrator.execute(task)
        else:
            result = await research_agent.execute(task)

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)

        # Extract reasoning trace if ReAct mode was used
        reasoning_trace = []
        if hasattr(result, "_reasoning_trace") and result._reasoning_trace:
            trace = result._reasoning_trace
            for step in trace.steps:
                reasoning_trace.append({
                    "step_type": step.step_type.value,
                    "content": step.content[:500] if step.content else "",
                    "confidence": step.confidence,
                    "timestamp": step.timestamp.isoformat(),
                })
            logger.info(
                "react_trace_extracted",
                steps=len(reasoning_trace),
                total_iterations=trace.total_iterations,
                final_confidence=trace.confidence,
            )

        # Extract data from result (handle ResearchFindings dataclass)
        data = result.data
        papers = []
        repositories = []
        models = []
        datasets = []
        web_results = []

        if data:
            # Handle ResearchFindings dataclass
            if hasattr(data, "papers"):
                papers = [
                    {"title": p.title, "url": p.url, "description": p.description, "metadata": p.metadata}
                    for p in data.papers
                ]
            if hasattr(data, "repositories"):
                repositories = [
                    {"title": r.title, "url": r.url, "description": r.description, "metadata": r.metadata}
                    for r in data.repositories
                ]
            if hasattr(data, "models"):
                models = [
                    {"title": m.title, "url": m.url, "description": m.description, "metadata": m.metadata}
                    for m in data.models
                ]
            if hasattr(data, "datasets"):
                datasets = [
                    {"title": d.title, "url": d.url, "description": d.description, "metadata": d.metadata}
                    for d in data.datasets
                ]
            if hasattr(data, "web_results"):
                web_results = [
                    {"title": w.title, "url": w.url, "description": w.description, "metadata": w.metadata}
                    for w in data.web_results
                ]

        # Build smart summary based on what was found
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

        # === MEMORY RETRIEVAL ===
        llm_router = getattr(request.app.state, "llm_router", None)
        enhanced_memory = getattr(request.app.state, "enhanced_memory_service", None)

        # Retrieve relevant past research from memory
        memory_context = await _retrieve_relevant_memory(
            enhanced_memory,
            body.query,
            current_user.id,
            top_k=3,
        )

        # === ITERATIVE REFINEMENT LOOP ===
        answer = ""
        confidence = 0.0
        refinement_attempts = 0
        current_query = body.query
        max_refinements = 2  # Maximum refinement attempts

        if body.include_synthesis and llm_router and (papers or repositories or models or web_results):
            for attempt in range(max_refinements + 1):
                # Synthesize answer
                answer = await _synthesize_answer(
                    llm_router,
                    current_query,
                    papers[:5],
                    repositories[:5],
                    models[:3],
                    web_results[:5],
                    memory_context=memory_context,
                )

                if not answer:
                    break

                # Critique the answer
                sources_found = {
                    "papers": len(papers),
                    "repositories": len(repositories),
                    "models": len(models),
                    "web_results": len(web_results),
                }
                critique = await _critique_answer(
                    llm_router,
                    body.query,
                    answer,
                    sources_found,
                )

                confidence = critique["confidence"]
                logger.info(
                    "answer_critique",
                    attempt=attempt,
                    confidence=confidence,
                    is_sufficient=critique["is_sufficient"],
                    assessment=critique.get("assessment", "")[:100],
                )

                # If confidence is high enough or no refined query, stop
                if confidence >= 0.7 or not critique.get("refined_query"):
                    break

                # Refine query and re-search
                refined_query = critique["refined_query"]
                logger.info(
                    "query_refinement",
                    original=current_query,
                    refined=refined_query,
                    attempt=attempt + 1,
                )

                # Re-search with refined query
                refined_task = AgentTask(
                    type="research",
                    params={
                        "query": refined_query,
                        "source": "all",
                        "sources": body.sources,
                        "max_results": body.max_results,
                    },
                    user_id=current_user.id,
                )

                if orchestrator:
                    refined_result = await orchestrator.execute(refined_task)
                else:
                    refined_result = await research_agent.execute(refined_task)

                if refined_result.success and refined_result.data:
                    # Merge new results with existing (prioritize new)
                    new_data = refined_result.data
                    if hasattr(new_data, "papers"):
                        new_papers = [
                            {"title": p.title, "url": p.url, "description": p.description, "metadata": p.metadata}
                            for p in new_data.papers
                        ]
                        papers = new_papers + papers  # New results first
                        papers = papers[:body.max_results]
                    if hasattr(new_data, "repositories"):
                        new_repos = [
                            {"title": r.title, "url": r.url, "description": r.description, "metadata": r.metadata}
                            for r in new_data.repositories
                        ]
                        repositories = new_repos + repositories
                        repositories = repositories[:body.max_results]
                    if hasattr(new_data, "web_results"):
                        new_web = [
                            {"title": w.title, "url": w.url, "description": w.description, "metadata": w.metadata}
                            for w in new_data.web_results
                        ]
                        web_results = new_web + web_results
                        web_results = web_results[:body.max_results]

                current_query = refined_query
                refinement_attempts += 1

            # Log final synthesis result
            logger.info(
                "answer_synthesized",
                query=body.query,
                answer_length=len(answer),
                confidence=confidence,
                refinement_attempts=refinement_attempts,
            )

            # Store in memory based on confidence
            await _store_research_memory(
                enhanced_memory,
                body.query,
                answer,
                confidence,
                current_user.id,
            )

        # Update summary with refinement info
        if refinement_attempts > 0:
            summary = f"{summary} (refined {refinement_attempts}x)"

        # Extract debate results if present (from collaborative mode or auto-detected evaluative queries)
        debate_data = {}
        if data and hasattr(data, "get") and callable(data.get):
            debate_data = data.get("debate", {})
        elif isinstance(data, dict):
            debate_data = data.get("debate", {})

        # Check if evaluative query was auto-detected
        evaluative_auto_detected = False
        if body.reasoning_mode != "collaborative" and debate_data:
            evaluative_auto_detected = True

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
            insights=[],
            recommendations=[],
            confidence=confidence,
            refinement_attempts=refinement_attempts,
            reasoning_trace=reasoning_trace,
            debate=debate_data,
            metadata={
                "execution_time_ms": result.execution_time_ms,
                "sources_searched": body.sources,
                "reasoning_mode": body.reasoning_mode,
                "memory_matches": len(memory_context),
                "memory_used": len(memory_context) > 0,
                "evaluative_auto_detected": evaluative_auto_detected,
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
