"""Memory API schemas."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """A memory entry from AgentCore."""

    id: str
    type: Literal["short_term", "long_term", "knowledge"]
    content: dict[str, Any]
    created_at: datetime
    expires_at: datetime | None = None
    summary_text: str = ""


class MemorySearchRequest(BaseModel):
    """Request for searching memory."""

    query: str = Field(default="", max_length=500)
    memory_type: Literal["short_term", "long_term", "all"] = "all"
    max_results: int = Field(default=10, ge=1, le=100)


class MemorySearchResponse(BaseModel):
    """Response for memory search."""

    entries: list[MemoryEntry]
    total_count: int


class PromoteToKnowledgeRequest(BaseModel):
    """Request to promote memory to knowledge."""

    topic: str = Field(..., min_length=1, max_length=200)
    summary: str = Field(..., min_length=1, max_length=2000)
    session_ids: list[str] = Field(default_factory=list)


class KnowledgeItem(BaseModel):
    """A knowledge item."""

    id: str
    topic: str
    summary: str
    source_sessions: list[str]
    promoted_at: datetime
