"""Subscription API schemas."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


SubscriptionTypeEnum = Literal[
    "arxiv_category",
    "arxiv_author",
    "arxiv_keyword",
    "hf_task",
    "hf_author",
    "github_repo",
    "github_topic",
]


class SubscriptionSchema(BaseModel):
    """Subscription representation."""

    id: str = Field(..., description="Subscription ID")
    type: SubscriptionTypeEnum = Field(..., description="Subscription type")
    value: str = Field(..., description="Subscribed value (category, author, keyword, etc.)")
    enabled: bool = Field(default=True, description="Whether subscription is active")
    created_at: datetime = Field(..., description="Creation timestamp")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class CreateSubscriptionRequest(BaseModel):
    """Request to create a subscription."""

    type: SubscriptionTypeEnum = Field(..., description="Subscription type")
    value: str = Field(
        ..., min_length=1, max_length=200, description="Value to subscribe to"
    )
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class UpdateSubscriptionRequest(BaseModel):
    """Request to update a subscription."""

    enabled: bool | None = Field(default=None, description="Enable/disable subscription")
    metadata: dict | None = Field(default=None, description="Update metadata")


class RecommendationSchema(BaseModel):
    """A content recommendation."""

    content_id: str = Field(..., description="Unique content identifier")
    source: str = Field(..., description="Content source (arxiv, huggingface, github)")
    title: str = Field(..., description="Content title")
    url: str = Field(..., description="Content URL")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    matched_tags: list[str] = Field(default_factory=list, description="Matching interest tags")
    reason: str = Field(default="", description="Human-readable match reason")


class RecommendationListResponse(BaseModel):
    """Response for recommendation list."""

    recommendations: list[RecommendationSchema] = Field(
        default_factory=list, description="List of recommendations"
    )
    total: int = Field(default=0, description="Total recommendations available")
    page: int = Field(default=1, description="Current page")
    page_size: int = Field(default=10, description="Items per page")


class DeliveryPreferencesSchema(BaseModel):
    """User's delivery preferences."""

    max_daily: int = Field(
        default=10, ge=1, le=100, description="Max daily recommendations"
    )
    min_score: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Minimum relevance score"
    )
    batch_size: int = Field(
        default=5, ge=1, le=20, description="Max items per notification"
    )
    cooldown_hours: int = Field(
        default=4, ge=1, le=24, description="Hours between notifications"
    )
    channels: list[str] = Field(
        default_factory=lambda: ["in_app"], description="Delivery channels"
    )


class UpdateDeliveryPreferencesRequest(BaseModel):
    """Request to update delivery preferences."""

    max_daily: int | None = Field(default=None, ge=1, le=100)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    batch_size: int | None = Field(default=None, ge=1, le=20)
    cooldown_hours: int | None = Field(default=None, ge=1, le=24)
    channels: list[str] | None = Field(default=None)
