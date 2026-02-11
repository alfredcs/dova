"""
Chat Schemas for DOVA API.
"""

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: float | None = Field(default=None, description="Unix timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ChatRequest(BaseModel):
    """Request schema for chat messages."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User message",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for continuing a conversation (omit for new session)",
    )
    sources: list[str] = Field(
        default=["arxiv", "github", "huggingface", "web"],
        description="Sources to search when research is needed",
    )
    show_thinking: bool = Field(
        default=False,
        description="Include thinking steps in response",
    )
    reasoning_mode: str = Field(
        default="standard",
        description="Reasoning mode: standard, react, collaborative, deep",
    )
    auto_debate: bool = Field(
        default=True,
        description="Auto-detect evaluative queries and trigger debate",
    )
    enable_two_pass: bool = Field(
        default=True,
        description="Enable two-pass emergent insight generation (deep mode)",
    )
    orchestrator: str = Field(
        default="standard",
        description="Orchestrator type: standard (task-graph) or thinking (deliberation-first)",
    )


class ThinkingStep(BaseModel):
    """A step in the chain-of-thought reasoning."""

    step_type: str = Field(..., description="Type: observation, reasoning, plan, action, reflection")
    content: str = Field(..., description="Step content")


class ImageResult(BaseModel):
    """Generated image result."""

    url: str = Field(..., description="URL or path to the generated image")
    prompt: str = Field(..., description="Prompt used to generate the image")
    resolution: str = Field(default="1024x1024", description="Image resolution")
    seed: int = Field(default=0, description="Seed used for generation")


class ChatResponse(BaseModel):
    """Response schema for chat messages."""

    session_id: str = Field(..., description="Session ID for continuing the conversation")
    message: str = Field(..., description="Assistant response")
    thinking: list[ThinkingStep] = Field(
        default_factory=list,
        description="Chain-of-thought reasoning steps (if show_thinking=true)",
    )
    action_taken: str | None = Field(
        default=None,
        description="Action taken (research, debate, synthesize, respond, image)",
    )
    sources_used: list[str] = Field(
        default_factory=list,
        description="Sources that were searched",
    )
    research_results: dict[str, Any] | None = Field(
        default=None,
        description="Research results if research action was taken",
    )
    debate_results: dict[str, Any] | None = Field(
        default=None,
        description="Debate results if debate action was taken",
    )
    images: list[ImageResult] = Field(
        default_factory=list,
        description="Generated images if image action was taken",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Response metadata",
    )


class SessionInfo(BaseModel):
    """Information about a chat session."""

    session_id: str = Field(..., description="Session ID")
    created_at: float = Field(..., description="Unix timestamp of creation")
    last_activity: float = Field(..., description="Unix timestamp of last activity")
    turn_count: int = Field(..., description="Number of conversation turns")
    topic: str = Field(default="", description="Main topic being discussed")


class SessionListResponse(BaseModel):
    """Response for listing chat sessions."""

    sessions: list[SessionInfo] = Field(default_factory=list, description="List of sessions")
    total: int = Field(default=0, description="Total number of sessions")
