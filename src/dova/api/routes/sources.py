"""Source management API endpoints."""
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from dova.api.middleware.auth import User, get_current_user
from dova.api.schemas.sources import (
    CreateSourceRequest, UpdateSourceRequest, RecordInteractionRequest,
    SourceSchema, QualityMetricsSchema,
)
from dova.services.source_types import SourceConfig, SourceType

router = APIRouter(prefix="/sources")
logger = structlog.get_logger(__name__)


@router.get("", response_model=list[SourceSchema])
async def list_sources(
    request: Request,
    enabled_only: bool = False,
    current_user: User = Depends(get_current_user),
) -> list[SourceSchema]:
    """List all sources for the current user."""
    registry = getattr(request.app.state, "source_registry", None)
    if not registry:
        return []

    sources = await registry.get_sources(current_user.id, enabled_only=enabled_only)
    return [
        SourceSchema(
            id=s.id, name=s.name, source_type=s.source_type.value,
            enabled=s.enabled,
            config=None,  # Don't expose auth values
            quality=QualityMetricsSchema(
                query_count=s.quality.query_count,
                click_count=s.quality.click_count,
                save_count=s.quality.save_count,
                quality_score=s.quality.quality_score,
            ),
            created_at=s.created_at,
        )
        for s in sources
    ]


@router.post("", response_model=SourceSchema)
async def create_source(
    request: Request,
    body: CreateSourceRequest,
    current_user: User = Depends(get_current_user),
) -> SourceSchema:
    """Add a custom source."""
    registry = getattr(request.app.state, "source_registry", None)
    if not registry:
        raise HTTPException(status_code=503, detail="Source service not available")

    config = SourceConfig(
        url=body.config.url,
        headers=body.config.headers,
        auth_type=body.config.auth_type,
        auth_value=body.config.auth_value,
        refresh_interval_minutes=body.config.refresh_interval_minutes,
        content_selector=body.config.content_selector,
    )

    source = await registry.add_source(
        user_id=current_user.id,
        name=body.name,
        source_type=SourceType(body.source_type),
        config=config,
    )

    return SourceSchema(
        id=source.id, name=source.name, source_type=source.source_type.value,
        enabled=source.enabled, config=None,
        quality=QualityMetricsSchema(quality_score=0.5),
        created_at=source.created_at,
    )


@router.put("/{source_id}", response_model=SourceSchema)
async def update_source(
    request: Request,
    source_id: str,
    body: UpdateSourceRequest,
    current_user: User = Depends(get_current_user),
) -> SourceSchema:
    """Update a source."""
    registry = getattr(request.app.state, "source_registry", None)
    if not registry:
        raise HTTPException(status_code=503, detail="Source service not available")

    updates = {}
    if body.name is not None:
        updates["name"] = body.name
    if body.enabled is not None:
        updates["enabled"] = body.enabled

    source = await registry.update_source(current_user.id, source_id, **updates)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    return SourceSchema(
        id=source.id, name=source.name, source_type=source.source_type.value,
        enabled=source.enabled, config=None,
        quality=QualityMetricsSchema(
            query_count=source.quality.query_count,
            click_count=source.quality.click_count,
            save_count=source.quality.save_count,
            quality_score=source.quality.quality_score,
        ),
        created_at=source.created_at,
    )


@router.delete("/{source_id}")
async def delete_source(
    request: Request,
    source_id: str,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Delete a custom source."""
    registry = getattr(request.app.state, "source_registry", None)
    if not registry:
        raise HTTPException(status_code=503, detail="Source service not available")

    deleted = await registry.delete_source(current_user.id, source_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Source not found or cannot be deleted")

    return {"status": "deleted", "id": source_id}


@router.post("/interact")
async def record_interaction(
    request: Request,
    body: RecordInteractionRequest,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Record an implicit quality signal (click, save, etc.)."""
    registry = getattr(request.app.state, "source_registry", None)
    if not registry:
        return {"status": "skipped"}

    await registry.record_interaction(
        user_id=current_user.id,
        source_id=body.source_id,
        interaction_type=body.interaction_type,
        result_position=body.result_position,
        result_count=body.result_count,
    )

    return {"status": "recorded"}
