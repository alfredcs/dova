"""Source API schemas."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SourceConfigSchema(BaseModel):
    """Configuration for a custom source."""
    url: str = Field(..., min_length=1, description="Source URL")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    auth_type: Literal["bearer", "api_key", "basic"] | None = Field(
        default=None, description="Authentication type"
    )
    auth_value: str | None = Field(default=None, description="Authentication value")
    refresh_interval_minutes: int = Field(
        default=60, ge=1, le=1440, description="Refresh interval in minutes"
    )
    content_selector: str | None = Field(
        default=None, description="CSS selector for web scraping"
    )


class QualityMetricsSchema(BaseModel):
    """Quality metrics for a source."""
    query_count: int = Field(default=0, description="Number of queries")
    click_count: int = Field(default=0, description="Number of clicks")
    save_count: int = Field(default=0, description="Number of saves")
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Quality score")


class SourceSchema(BaseModel):
    """Source representation."""
    id: str = Field(..., description="Source ID")
    name: str = Field(..., description="Source name")
    source_type: Literal["builtin", "web_url", "rss_feed", "api"] = Field(
        ..., description="Source type"
    )
    enabled: bool = Field(default=True, description="Whether source is enabled")
    config: SourceConfigSchema | None = Field(default=None, description="Source config")
    quality: QualityMetricsSchema = Field(
        default_factory=QualityMetricsSchema, description="Quality metrics"
    )
    created_at: datetime = Field(..., description="Creation timestamp")


class CreateSourceRequest(BaseModel):
    """Request to create a custom source."""
    name: str = Field(..., min_length=1, max_length=100, description="Source name")
    source_type: Literal["web_url", "rss_feed", "api"] = Field(
        ..., description="Source type"
    )
    config: SourceConfigSchema = Field(..., description="Source configuration")


class UpdateSourceRequest(BaseModel):
    """Request to update a source."""
    name: str | None = Field(default=None, description="New name")
    enabled: bool | None = Field(default=None, description="Enable/disable source")
    config: SourceConfigSchema | None = Field(default=None, description="New config")


class RecordInteractionRequest(BaseModel):
    """Request to record an interaction for quality tracking."""
    source_id: str = Field(..., description="Source ID")
    interaction_type: Literal["query", "click", "save"] = Field(
        ..., description="Type of interaction"
    )
    result_position: int | None = Field(default=None, description="Position of result clicked")
    result_count: int = Field(default=0, description="Number of results returned")
