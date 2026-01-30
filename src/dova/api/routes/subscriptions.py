"""Subscription and recommendation API endpoints."""
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from dova.api.middleware.auth import User, get_current_user
from dova.api.schemas.subscriptions import (
    CreateSubscriptionRequest,
    DeliveryPreferencesSchema,
    RecommendationListResponse,
    RecommendationSchema,
    SubscriptionSchema,
    UpdateDeliveryPreferencesRequest,
    UpdateSubscriptionRequest,
)
from dova.services.recommendation.subscriptions import SubscriptionType

router = APIRouter(prefix="/subscriptions")
logger = structlog.get_logger(__name__)


@router.get("", response_model=list[SubscriptionSchema])
async def list_subscriptions(
    request: Request,
    enabled_only: bool = Query(default=True, description="Only return enabled subscriptions"),
    current_user: User = Depends(get_current_user),
) -> list[SubscriptionSchema]:
    """List all subscriptions for the current user."""
    manager = getattr(request.app.state, "subscription_manager", None)
    if not manager:
        return []

    subs = await manager.list_subscriptions(current_user.id, enabled_only=enabled_only)
    return [
        SubscriptionSchema(
            id=str(s.id),
            type=s.type.value,
            value=s.value,
            enabled=s.enabled,
            created_at=s.created_at,
            metadata=s.metadata,
        )
        for s in subs
    ]


@router.post("", response_model=SubscriptionSchema, status_code=201)
async def create_subscription(
    request: Request,
    body: CreateSubscriptionRequest,
    current_user: User = Depends(get_current_user),
) -> SubscriptionSchema:
    """Create a new subscription."""
    manager = getattr(request.app.state, "subscription_manager", None)
    if not manager:
        raise HTTPException(status_code=503, detail="Subscription service not available")

    try:
        sub_type = SubscriptionType(body.type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid subscription type: {body.type}")

    subscription = await manager.create(
        user_id=current_user.id,
        sub_type=sub_type,
        value=body.value,
        metadata=body.metadata,
    )

    logger.info(
        "subscription_created",
        user_id=current_user.id,
        type=body.type,
        value=body.value,
    )

    return SubscriptionSchema(
        id=str(subscription.id),
        type=subscription.type.value,
        value=subscription.value,
        enabled=subscription.enabled,
        created_at=subscription.created_at,
        metadata=subscription.metadata,
    )


@router.get("/{sub_id}", response_model=SubscriptionSchema)
async def get_subscription(
    request: Request,
    sub_id: str,
    current_user: User = Depends(get_current_user),
) -> SubscriptionSchema:
    """Get a specific subscription."""
    manager = getattr(request.app.state, "subscription_manager", None)
    if not manager:
        raise HTTPException(status_code=503, detail="Subscription service not available")

    try:
        sub_uuid = UUID(sub_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid subscription ID format")

    subscription = await manager.get(current_user.id, sub_uuid)
    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")

    return SubscriptionSchema(
        id=str(subscription.id),
        type=subscription.type.value,
        value=subscription.value,
        enabled=subscription.enabled,
        created_at=subscription.created_at,
        metadata=subscription.metadata,
    )


@router.patch("/{sub_id}", response_model=SubscriptionSchema)
async def update_subscription(
    request: Request,
    sub_id: str,
    body: UpdateSubscriptionRequest,
    current_user: User = Depends(get_current_user),
) -> SubscriptionSchema:
    """Update a subscription (enable/disable)."""
    manager = getattr(request.app.state, "subscription_manager", None)
    if not manager:
        raise HTTPException(status_code=503, detail="Subscription service not available")

    try:
        sub_uuid = UUID(sub_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid subscription ID format")

    if body.enabled is not None:
        subscription = await manager.toggle(current_user.id, sub_uuid, body.enabled)
    else:
        subscription = await manager.get(current_user.id, sub_uuid)

    if not subscription:
        raise HTTPException(status_code=404, detail="Subscription not found")

    return SubscriptionSchema(
        id=str(subscription.id),
        type=subscription.type.value,
        value=subscription.value,
        enabled=subscription.enabled,
        created_at=subscription.created_at,
        metadata=subscription.metadata,
    )


@router.delete("/{sub_id}")
async def delete_subscription(
    request: Request,
    sub_id: str,
    current_user: User = Depends(get_current_user),
) -> dict:
    """Delete a subscription."""
    manager = getattr(request.app.state, "subscription_manager", None)
    if not manager:
        raise HTTPException(status_code=503, detail="Subscription service not available")

    try:
        sub_uuid = UUID(sub_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid subscription ID format")

    deleted = await manager.delete(current_user.id, sub_uuid)
    if not deleted:
        raise HTTPException(status_code=404, detail="Subscription not found")

    logger.info("subscription_deleted", user_id=current_user.id, sub_id=sub_id)
    return {"status": "deleted", "id": sub_id}


# Recommendations endpoints


@router.get("/recommendations", response_model=RecommendationListResponse)
async def get_recommendations(
    request: Request,
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=10, ge=1, le=50, description="Items per page"),
    current_user: User = Depends(get_current_user),
) -> RecommendationListResponse:
    """Get personalized recommendations for the current user."""
    delivery_manager = getattr(request.app.state, "delivery_manager", None)
    if not delivery_manager:
        return RecommendationListResponse(recommendations=[], total=0, page=page, page_size=page_size)

    # Get pending recommendations
    pending = await delivery_manager.get_pending_recommendations(
        current_user.id,
        limit=page_size * page,
    )

    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    page_items = pending[start:end]

    recommendations = [
        RecommendationSchema(
            content_id=item.get("content_id", ""),
            source=item.get("source", ""),
            title=item.get("title", ""),
            url=item.get("url", ""),
            score=item.get("score", 0.0),
            matched_tags=item.get("matched_tags", []),
            reason=item.get("reason", ""),
        )
        for item in page_items
    ]

    return RecommendationListResponse(
        recommendations=recommendations,
        total=len(pending),
        page=page,
        page_size=page_size,
    )


# Delivery preferences endpoints


@router.get("/preferences", response_model=DeliveryPreferencesSchema)
async def get_delivery_preferences(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> DeliveryPreferencesSchema:
    """Get user's recommendation delivery preferences."""
    delivery_manager = getattr(request.app.state, "delivery_manager", None)
    if not delivery_manager:
        return DeliveryPreferencesSchema()

    prefs = await delivery_manager._get_preferences(current_user.id)
    return DeliveryPreferencesSchema(
        max_daily=prefs.max_daily,
        min_score=prefs.min_score,
        batch_size=prefs.batch_size,
        cooldown_hours=prefs.cooldown_hours,
        channels=prefs.channels,
    )


@router.patch("/preferences", response_model=DeliveryPreferencesSchema)
async def update_delivery_preferences(
    request: Request,
    body: UpdateDeliveryPreferencesRequest,
    current_user: User = Depends(get_current_user),
) -> DeliveryPreferencesSchema:
    """Update user's recommendation delivery preferences."""
    delivery_manager = getattr(request.app.state, "delivery_manager", None)
    if not delivery_manager:
        raise HTTPException(status_code=503, detail="Delivery service not available")

    prefs = await delivery_manager.update_preferences(
        current_user.id,
        max_daily=body.max_daily,
        min_score=body.min_score,
        batch_size=body.batch_size,
        cooldown_hours=body.cooldown_hours,
        channels=body.channels,
    )

    logger.info("delivery_preferences_updated", user_id=current_user.id)
    return DeliveryPreferencesSchema(
        max_daily=prefs.max_daily,
        min_score=prefs.min_score,
        batch_size=prefs.batch_size,
        cooldown_hours=prefs.cooldown_hours,
        channels=prefs.channels,
    )
