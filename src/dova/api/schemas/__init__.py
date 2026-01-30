"""DOVA API Schemas."""

from dova.api.schemas.common import UserPreferencesSchema
from dova.api.schemas.memory import (
    KnowledgeItem,
    MemoryEntry,
    MemorySearchRequest,
    MemorySearchResponse,
    PromoteToKnowledgeRequest,
)
from dova.api.schemas.research import (
    ResearchRequest,
    ResearchResponse,
    SearchRequest,
    SearchResponse,
)
from dova.api.schemas.subscriptions import (
    CreateSubscriptionRequest,
    DeliveryPreferencesSchema,
    RecommendationListResponse,
    RecommendationSchema,
    SubscriptionSchema,
    UpdateDeliveryPreferencesRequest,
    UpdateSubscriptionRequest,
)

__all__ = [
    "KnowledgeItem",
    "MemoryEntry",
    "MemorySearchRequest",
    "MemorySearchResponse",
    "PromoteToKnowledgeRequest",
    "UserPreferencesSchema",
    "ResearchRequest",
    "ResearchResponse",
    "SearchRequest",
    "SearchResponse",
    "CreateSubscriptionRequest",
    "DeliveryPreferencesSchema",
    "RecommendationListResponse",
    "RecommendationSchema",
    "SubscriptionSchema",
    "UpdateDeliveryPreferencesRequest",
    "UpdateSubscriptionRequest",
]
