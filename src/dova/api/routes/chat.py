"""
Chat Endpoints for DOVA API.

Provides multi-turn conversational interface similar to Claude.ai/ChatGPT.
"""

import asyncio
import json
import time
from typing import Any

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from dova.api.middleware.auth import User, get_current_user
from dova.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ImageResult,
    SessionInfo,
    SessionListResponse,
    ThinkingStep,
)
from dova.cli.interact import InteractiveSession

router = APIRouter()
logger = structlog.get_logger(__name__)

# In-memory session storage (in production, use Redis)
_sessions: dict[str, InteractiveSession] = {}


def _get_or_create_session(
    session_id: str | None,
    user_id: str,
    settings: Any,
    llm_router: Any,
    mcp_client: Any,
    memory_service: Any,
    orchestrator_type: str = "standard",
    orchestrator: Any = None,
) -> tuple[InteractiveSession, str, bool]:
    """Get existing session or create new one."""
    is_new = False

    if session_id and session_id in _sessions:
        session = _sessions[session_id]
        # Verify session belongs to this user
        if session.user_id != user_id:
            raise HTTPException(status_code=403, detail="Session belongs to another user")
        return session, session_id, is_new

    # Create new session
    session = InteractiveSession(
        user_id=user_id,
        show_thinking=True,  # Always track thinking internally
        verbose=False,
        orchestrator_type=orchestrator_type,
    )

    # Inject dependencies
    session._llm_router = llm_router
    session._mcp_client = mcp_client
    session._memory_service = memory_service
    session._settings = settings

    # Inject shared orchestrator only for thinking mode sessions
    if orchestrator is not None and orchestrator_type == "thinking":
        session._thinking_orchestrator = orchestrator

    # Start the session
    new_session_id = session.start_session()
    _sessions[new_session_id] = session
    is_new = True

    logger.info("chat_session_created", session_id=new_session_id, user_id=user_id, orchestrator=orchestrator_type)
    return session, new_session_id, is_new


@router.post("/chat", response_model=ChatResponse)
async def send_message(
    request: Request,
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    """
    Send a message and receive a response in a multi-turn conversation.

    Features:
    - Automatic session management (creates new session if not provided)
    - Chain-of-thought reasoning
    - Automatic research when needed
    - Debate analysis for evaluative queries
    - Memory integration for context

    Args:
        body: Chat request with message and optional session_id

    Returns:
        Assistant response with session_id for continuing conversation
    """
    start_time = time.time()

    logger.info(
        "chat_request",
        user_id=current_user.id,
        session_id=body.session_id,
        message_preview=body.message[:100],
    )

    try:
        # Get dependencies from app state
        settings = getattr(request.app.state, "settings", None)
        llm_router = getattr(request.app.state, "llm_router", None)
        mcp_client = getattr(request.app.state, "mcp_client", None)
        memory_service = getattr(request.app.state, "enhanced_memory_service", None)
        orchestrator = getattr(request.app.state, "orchestrator", None)

        if not llm_router:
            raise HTTPException(status_code=503, detail="Chat service not available")

        # Get or create session
        session, session_id, is_new = _get_or_create_session(
            session_id=body.session_id,
            user_id=current_user.id,
            settings=settings,
            llm_router=llm_router,
            mcp_client=mcp_client,
            memory_service=memory_service,
            orchestrator_type=body.orchestrator,
            orchestrator=orchestrator,
        )

        # Configure session settings for this request
        session._research_sources = body.sources
        session._reasoning_mode = body.reasoning_mode
        session._auto_debate = body.auto_debate
        session._force_debate = (
            body.always_debate
            or body.reasoning_mode in ("collaborative", "deep")
        )
        session._enable_two_pass = body.enable_two_pass
        session.show_thinking = body.show_thinking

        # Process the message
        response_text = await session.process_input(body.message)

        # Extract thinking steps if requested
        thinking_steps = []
        if body.show_thinking and session.state and session.state.conversation:
            last_turn = session.state.conversation[-1]
            if last_turn.role == "assistant" and last_turn.thought_chain:
                thinking_steps = [
                    ThinkingStep(step_type=step.step_type, content=step.content)
                    for step in last_turn.thought_chain
                ]

        # Extract action and results from last turn
        action_taken = None
        research_results = None
        debate_results = None
        images: list[ImageResult] = []
        sources_used = []
        intent_weights: dict[str, float] = {}

        if session.state and session.state.conversation:
            last_turn = session.state.conversation[-1]
            if last_turn.role == "assistant":
                action_taken = last_turn.action_taken
                if last_turn.action_result:
                    result = last_turn.action_result
                    intent_weights = result.get("intent_weights") or {}
                    # ThinkingOrchestrator packs debate output into the same
                    # action_result dict as research, even when action_taken
                    # is "use_tools". Surface it whenever present.
                    if result.get("bull_strengths") or result.get("bear_concerns"):
                        debate_results = {
                            "summary": result.get("debate_summary", "") or result.get("summary", ""),
                            "bull_strengths": result.get("bull_strengths", []),
                            "bear_concerns": result.get("bear_concerns", []),
                            "recommendation": result.get("recommendation", ""),
                            "confidence": result.get("confidence_score", result.get("confidence", 0)),
                        }
                    # Derive top_papers + pubmed_papers the same way as the
                    # streaming path and /api/v1/research so the UI receives a
                    # consistent shape regardless of which endpoint it called.
                    from dova.agents.thinking_orchestrator import ThinkingOrchestrator
                    research_data = ThinkingOrchestrator.extract_research_data(
                        {"action_result": result, "deliberation": {"intent_weights": intent_weights}}
                    )

                    def _build_research_payload(answer_src: str) -> dict[str, Any]:
                        return {
                            "papers": research_data.get("papers", []),
                            "pubmed_papers": research_data.get("pubmed_papers", []),
                            "top_papers": research_data.get("top_papers", []),
                            "cross_domain_bridges": research_data.get("cross_domain_bridges", []),
                        "drug_story": research_data.get("drug_story"),
                            "repositories": research_data.get("repositories", []),
                            "models": research_data.get("models", []),
                            "web_results": research_data.get("web_results", []),
                            "summary": result.get("summary", ""),
                            "answer": answer_src,
                        }

                    if action_taken == "research":
                        research_results = _build_research_payload(result.get("answer", ""))
                        sources_used = body.sources
                    elif action_taken == "debate":
                        debate_results = {
                            "summary": result.get("summary", ""),
                            "bull_strengths": result.get("bull_strengths", []),
                            "bear_concerns": result.get("bear_concerns", []),
                            "recommendation": result.get("recommendation", ""),
                            "confidence": result.get("confidence", 0),
                        }
                        # Collaborative/deep mode includes research data alongside debate
                        if any(
                            research_data.get(k)
                            for k in ("papers", "pubmed_papers", "top_papers",
                                      "repositories", "models", "web_results")
                        ):
                            research_results = _build_research_payload(result.get("answer", ""))
                            sources_used = body.sources

                    # Extract images from action_result (ThinkingOrchestrator returns them)
                    raw_images = result.get("images", [])
                    for img in raw_images:
                        if isinstance(img, dict):
                            images.append(ImageResult(
                                url=img.get("url", ""),
                                prompt=img.get("prompt", ""),
                                resolution=img.get("resolution", "1024x1024"),
                                seed=img.get("seed", 0),
                            ))

        execution_time = time.time() - start_time

        logger.info(
            "chat_response",
            session_id=session_id,
            action=action_taken,
            response_length=len(response_text),
            execution_time_ms=int(execution_time * 1000),
        )

        return ChatResponse(
            session_id=session_id,
            message=response_text,
            thinking=thinking_steps,
            action_taken=action_taken,
            sources_used=sources_used,
            research_results=research_results,
            debate_results=debate_results,
            images=images,
            intent_weights=intent_weights,
            metadata={
                "is_new_session": is_new,
                "execution_time_ms": int(execution_time * 1000),
                "turn_count": len(session.state.conversation) // 2 if session.state else 0,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("chat_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/upload", response_model=ChatResponse)
async def send_message_with_files(
    request: Request,
    message: str = Form(...),
    session_id: str | None = Form(default=None),
    sources: str = Form(default='["arxiv","github","huggingface","web","bio"]'),
    show_thinking: bool = Form(default=False),
    reasoning_mode: str = Form(default="standard"),
    auto_debate: bool = Form(default=True),
    always_debate: bool = Form(default=False),
    enable_two_pass: bool = Form(default=True),
    orchestrator: str = Form(default="standard"),
    files: list[UploadFile] = File(default=[]),
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    """
    Send a chat message with optional file attachments.

    Accepts multipart form data with files (.txt, .pdf, .png).
    File contents are extracted and appended to the message.
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
        parsed_sources = ["arxiv", "github", "huggingface", "web", "bio"]

    # Process attached files and combine with message
    combined_message = message
    if files:
        file_parts = []
        for f in files:
            content = await process_uploaded_file(f)
            file_parts.append(f"[File: {f.filename}]\n{content}")
        attached = "\n\n".join(file_parts)
        combined_message = f"{message}\n\n--- Attached Files ---\n\n{attached}"

    # Build a ChatRequest-equivalent body and delegate to the same logic
    body = ChatRequest.model_construct(
        message=combined_message,
        session_id=session_id,
        sources=parsed_sources,
        show_thinking=show_thinking,
        reasoning_mode=reasoning_mode,
        auto_debate=auto_debate,
        always_debate=always_debate,
        enable_two_pass=enable_two_pass,
        orchestrator=orchestrator,
    )
    return await send_message(request, body, current_user)


@router.post("/chat/stream")
async def stream_message(
    request: Request,
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """
    Stream a chat response via Server-Sent Events.

    Emits these event types:
      - thinking:       { step_type, content }
      - stage:          { stage, message, ... }
      - tool_complete:  { tool, count, items, ... }
      - log:            { step, status, elapsed_ms, ... }
      - synthesis_token:{ token }
      - complete:       full ChatResponse payload
      - error:          { message }
    """
    from dova.agents.base import AgentTask

    settings = getattr(request.app.state, "settings", None)
    llm_router = getattr(request.app.state, "llm_router", None)
    mcp_client = getattr(request.app.state, "mcp_client", None)
    memory_service = getattr(request.app.state, "enhanced_memory_service", None)
    orchestrator = getattr(request.app.state, "orchestrator", None)

    if not llm_router:
        raise HTTPException(status_code=503, detail="Chat service not available")
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Streaming requires thinking orchestrator")

    session, session_id, is_new = _get_or_create_session(
        session_id=body.session_id,
        user_id=current_user.id,
        settings=settings,
        llm_router=llm_router,
        mcp_client=mcp_client,
        memory_service=memory_service,
        orchestrator_type="thinking",
        orchestrator=orchestrator,
    )
    # Log exactly what the client sent so we can diagnose UI regressions
    # (e.g., stale cached HTML dropping sources silently).
    logger.info(
        "chat_stream_request",
        sources_from_client=list(body.sources),
        reasoning_mode=body.reasoning_mode,
        orchestrator=body.orchestrator,
    )
    session._research_sources = body.sources
    session._reasoning_mode = body.reasoning_mode
    session._auto_debate = body.auto_debate
    session._enable_two_pass = body.enable_two_pass
    session.show_thinking = body.show_thinking

    # Record the user turn synchronously so history stays coherent even if
    # the client disconnects mid-stream.
    from dova.cli.interact import ConversationTurn, ThoughtStep  # local import to avoid cycles

    session.state.conversation.append(
        ConversationTurn(role="user", content=body.message)
    )

    queue: asyncio.Queue = asyncio.Queue()
    thinking_collected: list[ThoughtStep] = []

    async def progress_cb(event_type: str, data: dict) -> None:
        if event_type == "thinking":
            thinking_collected.append(
                ThoughtStep(
                    step_type=data.get("step_type", "reasoning"),
                    content=data.get("content", ""),
                )
            )
        await queue.put((event_type, data))

    async def _runner() -> None:
        start_time = time.time()
        try:
            # Force bull/bear debate when:
            #   - user toggled "Always run bull/bear debate" in settings, OR
            #   - reasoning mode is "collaborative" or "deep" (both imply it).
            force_debate = (
                body.always_debate
                or body.reasoning_mode in ("collaborative", "deep")
            )
            task = AgentTask(
                type="query",
                params={
                    "query": body.message,
                    "session_id": session_id,
                    "sources": body.sources,
                    "auto_debate": body.auto_debate,
                    "force_debate": force_debate,
                },
                user_id=current_user.id,
            )
            result = await orchestrator.execute(task, progress=progress_cb)

            if not result.success:
                await queue.put(("error", {"message": result.error or "Unknown error"}))
                return

            data = result.data or {}
            response_text = data.get("response", "")
            deliberation = data.get("deliberation", {})
            action_result = data.get("action_result") or {}

            # Same derivation as /api/v1/research so the UI receives
            # top_papers (ai/bio split by intent weights) and pubmed_papers
            # (flattened from bio MCP results) for chat-driven research.
            from dova.agents.thinking_orchestrator import ThinkingOrchestrator
            research_data = ThinkingOrchestrator.extract_research_data(data)

            session.state.conversation.append(
                ConversationTurn(
                    role="assistant",
                    content=response_text,
                    thought_chain=thinking_collected,
                    action_taken=deliberation.get("action"),
                    action_result=action_result,
                )
            )
            session.state.context["last_query"] = body.message
            session.state.context["last_action"] = deliberation.get("action")
            session.state.context["turn_count"] = (
                len(session.state.conversation) // 2
            )

            research_results = None
            debate_results = None
            sources_used: list[str] = []
            action_taken = deliberation.get("action")

            if action_result:
                has_any = any(
                    research_data.get(k)
                    for k in ("papers", "pubmed_papers", "top_papers",
                              "repositories", "models", "web_results")
                )
                if has_any:
                    research_results = {
                        "papers": research_data.get("papers", []),
                        "pubmed_papers": research_data.get("pubmed_papers", []),
                        "top_papers": research_data.get("top_papers", []),
                        "cross_domain_bridges": research_data.get("cross_domain_bridges", []),
                        "drug_story": research_data.get("drug_story"),
                        "repositories": research_data.get("repositories", []),
                        "models": research_data.get("models", []),
                        "web_results": research_data.get("web_results", []),
                        "summary": action_result.get("summary", ""),
                        "answer": response_text,
                    }
                    sources_used = body.sources
                # Debate output (when ThinkingOrchestrator ran bull/bear pass).
                if action_result.get("bull_strengths") or action_result.get("bear_concerns"):
                    debate_results = {
                        "summary": action_result.get("debate_summary", ""),
                        "bull_strengths": action_result.get("bull_strengths", []),
                        "bear_concerns": action_result.get("bear_concerns", []),
                        "recommendation": action_result.get("recommendation", ""),
                        "confidence": action_result.get("confidence_score", 0),
                    }

            images: list[dict] = []
            for img in action_result.get("images", []) or []:
                if isinstance(img, dict):
                    images.append({
                        "url": img.get("url", ""),
                        "prompt": img.get("prompt", ""),
                        "resolution": img.get("resolution", "1024x1024"),
                        "seed": img.get("seed", 0),
                    })

            await queue.put((
                "complete",
                {
                    "session_id": session_id,
                    "message": response_text,
                    "thinking": [
                        {"step_type": t.step_type, "content": t.content}
                        for t in thinking_collected
                    ],
                    "action_taken": action_taken,
                    "sources_used": sources_used,
                    "research_results": research_results,
                    "debate_results": debate_results,
                    "images": images,
                    # Expose the weighted intent distribution so the UI can
                    # show users how the orchestrator framed their question.
                    "intent_weights": deliberation.get("intent_weights", {}),
                    "metadata": {
                        "is_new_session": is_new,
                        "execution_time_ms": int((time.time() - start_time) * 1000),
                        "turn_count": len(session.state.conversation) // 2,
                    },
                },
            ))
        except Exception as exc:
            logger.exception("chat_stream_error", error=str(exc))
            await queue.put(("error", {"message": str(exc)}))
        finally:
            await queue.put(None)

    async def event_generator():
        runner_task = asyncio.create_task(_runner())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                evt_type, evt_data = item
                yield (
                    f"event: {evt_type}\n"
                    f"data: {json.dumps(evt_data, default=str)}\n\n"
                )
        except asyncio.CancelledError:
            runner_task.cancel()
            raise

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/chat/sessions", response_model=SessionListResponse)
async def list_sessions(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> SessionListResponse:
    """
    List all chat sessions for the current user.

    Returns:
        List of session information
    """
    user_sessions = []

    for session_id, session in _sessions.items():
        if session.user_id == current_user.id and session.state:
            user_sessions.append(
                SessionInfo(
                    session_id=session_id,
                    created_at=session.state.started_at,
                    last_activity=session.state.conversation[-1].timestamp
                    if session.state.conversation
                    else session.state.started_at,
                    turn_count=len(session.state.conversation) // 2,
                    topic=session.state.last_topic or "",
                )
            )

    # Sort by last activity (most recent first)
    user_sessions.sort(key=lambda s: s.last_activity, reverse=True)

    return SessionListResponse(
        sessions=user_sessions,
        total=len(user_sessions),
    )


@router.get("/chat/sessions/{session_id}", response_model=SessionInfo)
async def get_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
) -> SessionInfo:
    """
    Get information about a specific chat session.

    Args:
        session_id: The session ID

    Returns:
        Session information
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Session belongs to another user")

    if not session.state:
        raise HTTPException(status_code=404, detail="Session state not found")

    return SessionInfo(
        session_id=session_id,
        created_at=session.state.started_at,
        last_activity=session.state.conversation[-1].timestamp
        if session.state.conversation
        else session.state.started_at,
        turn_count=len(session.state.conversation) // 2,
        topic=session.state.last_topic or "",
    )


@router.delete("/chat/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
) -> dict[str, str]:
    """
    Delete a chat session.

    Args:
        session_id: The session ID to delete

    Returns:
        Confirmation message
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Session belongs to another user")

    del _sessions[session_id]
    logger.info("chat_session_deleted", session_id=session_id, user_id=current_user.id)

    return {"message": f"Session {session_id} deleted"}


@router.get("/chat/sessions/{session_id}/history")
async def get_session_history(
    session_id: str,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Get the conversation history for a session.

    Args:
        session_id: The session ID

    Returns:
        Conversation history
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = _sessions[session_id]
    if session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Session belongs to another user")

    if not session.state:
        raise HTTPException(status_code=404, detail="Session state not found")

    messages = []
    for turn in session.state.conversation:
        messages.append({
            "role": turn.role,
            "content": turn.content,
            "timestamp": turn.timestamp,
            "action": turn.action_taken,
        })

    return {
        "session_id": session_id,
        "messages": messages,
        "topic": session.state.last_topic or "",
        "entities": list(session.state.entities_discussed.keys()),
    }
