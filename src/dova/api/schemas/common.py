"""
Common Schemas for DOVA API.
"""

from typing import Any

from pydantic import BaseModel, Field


class UserPreferencesSchema(BaseModel):
    """Schema for user preferences updates."""

    interests: list[str] | None = Field(
        default=None,
        description="List of research interests",
        examples=[["machine learning", "NLP", "transformers"]],
    )
    preferred_sources: list[str] | None = Field(
        default=None,
        description="Preferred research sources",
        examples=[["arxiv", "github", "huggingface"]],
    )
    expertise_level: str | None = Field(
        default=None,
        description="User expertise level",
        examples=["beginner", "intermediate", "advanced", "expert"],
    )
    output_format: str | None = Field(
        default=None,
        description="Preferred output format",
        examples=["summary", "detailed", "technical"],
    )
    notification_frequency: str | None = Field(
        default=None,
        description="Notification frequency preference",
        examples=["realtime", "daily", "weekly", "none"],
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    request_id: str | None = Field(None, description="Request ID for debugging")


class PaginationParams(BaseModel):
    """Pagination parameters."""

    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")


class PaginatedResponse(BaseModel):
    """Base model for paginated responses."""

    items: list[Any] = Field(default_factory=list, description="Response items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    has_more: bool = Field(..., description="Whether more pages exist")
