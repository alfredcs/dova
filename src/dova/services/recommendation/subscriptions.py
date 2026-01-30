"""
Subscription management for proactive recommendations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger(__name__)


class SubscriptionType(Enum):
    """Types of content subscriptions."""

    ARXIV_CATEGORY = "arxiv_category"
    ARXIV_AUTHOR = "arxiv_author"
    ARXIV_KEYWORD = "arxiv_keyword"
    HF_TASK = "hf_task"
    HF_AUTHOR = "hf_author"
    GITHUB_REPO = "github_repo"
    GITHUB_TOPIC = "github_topic"


@dataclass
class Subscription:
    """A user's content subscription."""

    id: UUID
    user_id: str
    type: SubscriptionType
    value: str  # The category, author, keyword, etc.
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "type": self.type.value,
            "value": self.value,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Subscription":
        """Deserialize from dictionary."""
        return cls(
            id=UUID(data["id"]),
            user_id=data["user_id"],
            type=SubscriptionType(data["type"]),
            value=data["value"],
            enabled=data.get("enabled", True),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


class SubscriptionManager:
    """
    Manages user subscriptions for proactive content delivery.
    """

    def __init__(self, memory_service: Any | None = None):
        self.memory_service = memory_service
        self._cache: dict[str, dict[UUID, Subscription]] = {}
        self._logger = logger.bind(service="subscription_manager")

    async def create(
        self,
        user_id: str,
        sub_type: SubscriptionType,
        value: str,
        metadata: dict[str, Any] | None = None,
    ) -> Subscription:
        """
        Create a new subscription.

        Args:
            user_id: User creating the subscription
            sub_type: Type of subscription
            value: The value to subscribe to (category, keyword, etc.)
            metadata: Additional metadata

        Returns:
            Created subscription
        """
        # Check for duplicates
        existing = await self.list_subscriptions(user_id)
        for sub in existing:
            if sub.type == sub_type and sub.value.lower() == value.lower():
                self._logger.info(
                    "subscription_exists",
                    user_id=user_id,
                    type=sub_type.value,
                    value=value,
                )
                return sub

        subscription = Subscription(
            id=uuid4(),
            user_id=user_id,
            type=sub_type,
            value=value,
            metadata=metadata or {},
        )

        # Add to cache
        if user_id not in self._cache:
            self._cache[user_id] = {}
        self._cache[user_id][subscription.id] = subscription

        # Persist
        await self._persist(subscription)

        self._logger.info(
            "subscription_created",
            user_id=user_id,
            sub_id=str(subscription.id),
            type=sub_type.value,
            value=value,
        )

        return subscription

    async def list_subscriptions(self, user_id: str, enabled_only: bool = True) -> list[Subscription]:
        """
        List user's subscriptions.

        Args:
            user_id: User ID
            enabled_only: Only return enabled subscriptions

        Returns:
            List of subscriptions
        """
        # Try cache first
        if user_id in self._cache:
            subs = list(self._cache[user_id].values())
        else:
            # Load from memory service
            subs = await self._load_user_subscriptions(user_id)
            self._cache[user_id] = {s.id: s for s in subs}

        if enabled_only:
            subs = [s for s in subs if s.enabled]

        return sorted(subs, key=lambda s: s.created_at, reverse=True)

    async def get(self, user_id: str, sub_id: UUID) -> Subscription | None:
        """Get a specific subscription."""
        if user_id in self._cache and sub_id in self._cache[user_id]:
            return self._cache[user_id][sub_id]

        subs = await self.list_subscriptions(user_id, enabled_only=False)
        for sub in subs:
            if sub.id == sub_id:
                return sub

        return None

    async def delete(self, user_id: str, sub_id: UUID) -> bool:
        """
        Delete a subscription.

        Returns:
            True if deleted, False if not found
        """
        sub = await self.get(user_id, sub_id)
        if not sub:
            return False

        # Remove from cache
        if user_id in self._cache and sub_id in self._cache[user_id]:
            del self._cache[user_id][sub_id]

        # Remove from storage
        await self._delete(user_id, sub_id)

        self._logger.info(
            "subscription_deleted",
            user_id=user_id,
            sub_id=str(sub_id),
        )

        return True

    async def toggle(self, user_id: str, sub_id: UUID, enabled: bool) -> Subscription | None:
        """Toggle subscription enabled state."""
        sub = await self.get(user_id, sub_id)
        if not sub:
            return None

        sub.enabled = enabled

        # Update cache
        if user_id in self._cache:
            self._cache[user_id][sub_id] = sub

        # Persist
        await self._persist(sub)

        self._logger.info(
            "subscription_toggled",
            user_id=user_id,
            sub_id=str(sub_id),
            enabled=enabled,
        )

        return sub

    async def get_subscribers_for_content(
        self,
        content_type: str,
        content_tags: list[str],
    ) -> list[str]:
        """
        Find users subscribed to content matching given attributes.

        Args:
            content_type: Source type (arxiv, huggingface, github)
            content_tags: Tags/categories/topics of the content

        Returns:
            List of user IDs with matching subscriptions
        """
        # Map content type to subscription types
        type_mapping = {
            "arxiv": [SubscriptionType.ARXIV_CATEGORY, SubscriptionType.ARXIV_KEYWORD],
            "huggingface": [SubscriptionType.HF_TASK, SubscriptionType.HF_AUTHOR],
            "github": [SubscriptionType.GITHUB_REPO, SubscriptionType.GITHUB_TOPIC],
        }

        relevant_types = type_mapping.get(content_type, [])
        if not relevant_types:
            return []

        matching_users: set[str] = set()

        # Check all cached users
        for user_id, subs in self._cache.items():
            for sub in subs.values():
                if not sub.enabled:
                    continue
                if sub.type not in relevant_types:
                    continue

                # Check if subscription value matches any tag
                sub_value_lower = sub.value.lower()
                for tag in content_tags:
                    if sub_value_lower in tag.lower() or tag.lower() in sub_value_lower:
                        matching_users.add(user_id)
                        break

        return list(matching_users)

    async def _persist(self, subscription: Subscription) -> None:
        """Persist subscription to storage."""
        if not self.memory_service:
            return

        try:
            await self.memory_service.store_subscription(
                subscription.user_id,
                subscription.to_dict(),
            )
        except Exception as e:
            self._logger.error("subscription_persist_error", error=str(e))

    async def _delete(self, user_id: str, sub_id: UUID) -> None:
        """Delete subscription from storage."""
        if not self.memory_service:
            return

        try:
            await self.memory_service.delete_subscription(user_id, str(sub_id))
        except Exception as e:
            self._logger.error("subscription_delete_error", error=str(e))

    async def _load_user_subscriptions(self, user_id: str) -> list[Subscription]:
        """Load subscriptions from storage."""
        if not self.memory_service:
            return []

        try:
            data = await self.memory_service.get_subscriptions(user_id)
            return [Subscription.from_dict(d) for d in data]
        except Exception as e:
            self._logger.error("subscriptions_load_error", user_id=user_id, error=str(e))
            return []
