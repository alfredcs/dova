"""
Delivery manager for batching and sending recommendations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import structlog

from dova.services.recommendation.matcher import Match

logger = structlog.get_logger(__name__)


@dataclass
class DeliveryPreferences:
    """User preferences for recommendation delivery."""

    user_id: str
    max_daily: int = 10
    min_score: float = 0.75
    batch_size: int = 5
    cooldown_hours: int = 4
    channels: list[str] = field(default_factory=lambda: ["in_app"])


@dataclass
class DeliveryRecord:
    """Record of delivered recommendations."""

    user_id: str
    content_ids: list[str]
    delivered_at: datetime
    channel: str


class DeliveryManager:
    """
    Manages recommendation delivery with batching and frequency capping.

    Prevents notification fatigue through configurable limits.
    """

    def __init__(
        self,
        memory_service: Any | None = None,
        default_max_daily: int = 10,
        default_cooldown_hours: int = 4,
    ):
        self.memory_service = memory_service
        self.default_max_daily = default_max_daily
        self.default_cooldown_hours = default_cooldown_hours
        self._delivery_history: dict[str, list[DeliveryRecord]] = {}
        self._preferences_cache: dict[str, DeliveryPreferences] = {}
        self._logger = logger.bind(service="delivery_manager")

    async def prepare_delivery(
        self,
        user_id: str,
        matches: list[Match],
    ) -> list[Match]:
        """
        Prepare matches for delivery, applying caps and filtering.

        Args:
            user_id: Target user
            matches: Potential matches to deliver

        Returns:
            Filtered list of matches to actually deliver
        """
        prefs = await self._get_preferences(user_id)

        # Check cooldown
        if not await self._check_cooldown(user_id, prefs):
            self._logger.debug("delivery_cooldown", user_id=user_id)
            return []

        # Check daily cap
        remaining = await self._get_remaining_quota(user_id, prefs)
        if remaining <= 0:
            self._logger.debug("daily_cap_reached", user_id=user_id)
            return []

        # Filter by minimum score
        filtered = [m for m in matches if m.score >= prefs.min_score]

        # Apply remaining quota
        filtered = filtered[:remaining]

        # Batch size limit
        filtered = filtered[: prefs.batch_size]

        # Deduplicate (don't send same content twice)
        delivered_ids = await self._get_delivered_ids(user_id)
        filtered = [m for m in filtered if m.content_id not in delivered_ids]

        self._logger.debug(
            "delivery_prepared",
            user_id=user_id,
            original=len(matches),
            filtered=len(filtered),
        )

        return filtered

    async def record_delivery(
        self,
        user_id: str,
        matches: list[Match],
        channel: str = "in_app",
    ) -> None:
        """Record that recommendations were delivered."""
        if not matches:
            return

        record = DeliveryRecord(
            user_id=user_id,
            content_ids=[m.content_id for m in matches],
            delivered_at=datetime.utcnow(),
            channel=channel,
        )

        if user_id not in self._delivery_history:
            self._delivery_history[user_id] = []
        self._delivery_history[user_id].append(record)

        # Persist to memory service
        if self.memory_service:
            try:
                await self._persist_delivery(record)
            except Exception as e:
                self._logger.warning("delivery_persist_error", error=str(e))

        self._logger.info(
            "delivery_recorded",
            user_id=user_id,
            count=len(matches),
            channel=channel,
        )

    async def get_pending_recommendations(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Get pending (undelivered) recommendations for a user.

        This is for pull-based delivery (user requests recommendations).
        """
        # TODO: Implement storage of pending recommendations
        return []

    async def _get_preferences(self, user_id: str) -> DeliveryPreferences:
        """Get user's delivery preferences."""
        if user_id in self._preferences_cache:
            return self._preferences_cache[user_id]

        prefs = DeliveryPreferences(
            user_id=user_id,
            max_daily=self.default_max_daily,
            cooldown_hours=self.default_cooldown_hours,
        )

        if self.memory_service:
            try:
                data = await self.memory_service.get_user_preferences(user_id)
                if data:
                    prefs.max_daily = data.get("max_daily", prefs.max_daily)
                    prefs.min_score = data.get("min_score", prefs.min_score)
                    prefs.batch_size = data.get("batch_size", prefs.batch_size)
                    prefs.cooldown_hours = data.get("cooldown_hours", prefs.cooldown_hours)
                    prefs.channels = data.get("channels", prefs.channels)
            except Exception as e:
                self._logger.warning("prefs_load_error", user_id=user_id, error=str(e))

        self._preferences_cache[user_id] = prefs
        return prefs

    async def _check_cooldown(self, user_id: str, prefs: DeliveryPreferences) -> bool:
        """Check if user is past their cooldown period."""
        history = self._delivery_history.get(user_id, [])
        if not history:
            return True

        last_delivery = history[-1].delivered_at
        cooldown_end = last_delivery + timedelta(hours=prefs.cooldown_hours)
        return datetime.utcnow() >= cooldown_end

    async def _get_remaining_quota(self, user_id: str, prefs: DeliveryPreferences) -> int:
        """Get remaining daily delivery quota."""
        history = self._delivery_history.get(user_id, [])
        today = datetime.utcnow().date()

        today_count = sum(
            len(r.content_ids) for r in history if r.delivered_at.date() == today
        )

        return max(0, prefs.max_daily - today_count)

    async def _get_delivered_ids(self, user_id: str) -> set[str]:
        """Get set of content IDs already delivered to user."""
        history = self._delivery_history.get(user_id, [])
        ids: set[str] = set()
        for record in history:
            ids.update(record.content_ids)
        return ids

    async def _persist_delivery(self, record: DeliveryRecord) -> None:
        """Persist delivery record to memory service."""
        if not self.memory_service:
            return

        await self.memory_service.store_delivery_record(
            record.user_id,
            {
                "content_ids": record.content_ids,
                "delivered_at": record.delivered_at.isoformat(),
                "channel": record.channel,
            },
        )

    async def update_preferences(
        self,
        user_id: str,
        max_daily: int | None = None,
        min_score: float | None = None,
        batch_size: int | None = None,
        cooldown_hours: int | None = None,
        channels: list[str] | None = None,
    ) -> DeliveryPreferences:
        """Update user's delivery preferences."""
        prefs = await self._get_preferences(user_id)

        if max_daily is not None:
            prefs.max_daily = max_daily
        if min_score is not None:
            prefs.min_score = min_score
        if batch_size is not None:
            prefs.batch_size = batch_size
        if cooldown_hours is not None:
            prefs.cooldown_hours = cooldown_hours
        if channels is not None:
            prefs.channels = channels

        self._preferences_cache[user_id] = prefs

        if self.memory_service:
            try:
                await self.memory_service.update_user_preferences(
                    user_id,
                    {
                        "max_daily": prefs.max_daily,
                        "min_score": prefs.min_score,
                        "batch_size": prefs.batch_size,
                        "cooldown_hours": prefs.cooldown_hours,
                        "channels": prefs.channels,
                    },
                )
            except Exception as e:
                self._logger.warning("prefs_save_error", user_id=user_id, error=str(e))

        return prefs
