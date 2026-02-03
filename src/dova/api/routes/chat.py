"""
Chat Endpoints for DOVA API.

Provides multi-turn conversational interface similar to Claude.ai/ChatGPT.
"""

import time
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from dova.api.middleware.auth import User, get_current_user
from dova.api.schemas.chat import (
    ChatRequest,
    ChatResponse,
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
    )

    # Inject dependencies
    session._llm_router = llm_router
    session._mcp_client = mcp_client
    session._memory_service = memory_service
    session._settings = settings

    # Start the session
    new_session_id = session.start_session()
    _sessions[new_session_id] = session
    is_new = True

    logger.info("chat_session_created", session_id=new_session_id, user_id=user_id)
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
        )

        # Configure session settings for this request
        session._research_sources = body.sources
        session._reasoning_mode = body.reasoning_mode
        session._auto_debate = body.auto_debate
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
        sources_used = []

        if session.state and session.state.conversation:
            last_turn = session.state.conversation[-1]
            if last_turn.role == "assistant":
                action_taken = last_turn.action_taken
                if last_turn.action_result:
                    result = last_turn.action_result
                    if action_taken == "research":
                        research_results = {
                            "papers": result.get("papers", []),
                            "repositories": result.get("repositories", []),
                            "models": result.get("models", []),
                            "web_results": result.get("web_results", []),
                            "summary": result.get("summary", ""),
                            "answer": result.get("answer", ""),
                        }
                        sources_used = body.sources
                    elif action_taken == "debate":
                        debate_results = {
                            "summary": result.get("summary", ""),
                            "bull_strengths": result.get("bull_strengths", []),
                            "bear_concerns": result.get("bear_concerns", []),
                            "recommendation": result.get("recommendation", ""),
                            "confidence": result.get("confidence", 0),
                        }

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
