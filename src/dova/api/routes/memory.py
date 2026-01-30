"""Memory API endpoints."""

from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from dova.api.middleware.auth import User, get_current_user
from dova.api.schemas.memory import (
    KnowledgeItem,
    MemoryEntry,
    MemorySearchResponse,
    PromoteToKnowledgeRequest,
)

router = APIRouter(prefix="/memory")
logger = structlog.get_logger(__name__)


@router.get("/search", response_model=MemorySearchResponse)
async def search_memory(
    request: Request,
    q: str = "",
    type: str = "all",
    max_results: int = 10,
    current_user: User = Depends(get_current_user),
) -> MemorySearchResponse:
    """Search memory entries."""
    memory_service = getattr(request.app.state, "memory_service", None)
    if not memory_service:
        return MemorySearchResponse(entries=[], total_count=0)

    entries = await memory_service.search_memory(
        f"user:{current_user.id}", max_results
    )
    return MemorySearchResponse(
        entries=[
            MemoryEntry(
                id=e.id,
                type=e.type,
                content=e.content,
                created_at=e.created_at,
                summary_text=e.summary_text,
            )
            for e in entries
        ],
        total_count=len(entries),
    )


@router.get("/history", response_model=MemorySearchResponse)
async def get_history(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> MemorySearchResponse:
    """Get memory history for current user."""
    memory_service = getattr(request.app.state, "memory_service", None)
    if not memory_service:
        return MemorySearchResponse(entries=[], total_count=0)

    entries = await memory_service.search_memory(f"user:{current_user.id}", 50)
    return MemorySearchResponse(
        entries=[
            MemoryEntry(
                id=e.id,
                type=e.type,
                content=e.content,
                created_at=e.created_at,
                summary_text=e.summary_text,
            )
            for e in entries
        ],
        total_count=len(entries),
    )


@router.get("/knowledge", response_model=list[KnowledgeItem])
async def get_knowledge(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> list[KnowledgeItem]:
    """Get crystallized knowledge for current user."""
    memory_service = getattr(request.app.state, "memory_service", None)
    if not memory_service:
        return []

    entries = await memory_service.search_memory(f"knowledge:{current_user.id}", 50)
    return [
        KnowledgeItem(
            id=e.id,
            topic=e.content.get("topic", ""),
            summary=e.content.get("summary", ""),
            source_sessions=e.content.get("source_sessions", []),
            promoted_at=e.created_at,
        )
        for e in entries
    ]


@router.post("/promote", response_model=KnowledgeItem)
async def promote_to_knowledge(
    request: Request,
    body: PromoteToKnowledgeRequest,
    current_user: User = Depends(get_current_user),
) -> KnowledgeItem:
    """Promote memory entries to permanent knowledge."""
    memory_service = getattr(request.app.state, "memory_service", None)
    if not memory_service:
        raise HTTPException(status_code=503, detail="Memory service not available")

    knowledge_id = await memory_service.promote_to_knowledge(
        current_user.id, body.topic, body.summary, body.session_ids
    )
    return KnowledgeItem(
        id=knowledge_id,
        topic=body.topic,
        summary=body.summary,
        source_sessions=body.session_ids,
        promoted_at=datetime.utcnow(),
    )


@router.delete("/{memory_id}")
async def delete_memory(
    request: Request,
    memory_id: str,
    current_user: User = Depends(get_current_user),
) -> dict[str, str]:
    """Delete a memory entry."""
    memory_service = getattr(request.app.state, "memory_service", None)
    if not memory_service:
        raise HTTPException(status_code=503, detail="Memory service not available")

    await memory_service.delete_memory(memory_id)
    return {"status": "deleted", "id": memory_id}
