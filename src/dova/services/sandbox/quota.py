"""
Quota management for sandbox execution.
"""

from datetime import datetime, timedelta
from typing import Any

import structlog

from dova.services.sandbox.types import SandboxQuota, SandboxTier

logger = structlog.get_logger(__name__)


class QuotaManager:
    """
    Manages user quotas for sandbox execution.

    Tracks CPU and GPU time usage with daily resets.
    """

    def __init__(
        self,
        memory_service: Any | None = None,
        default_cpu_seconds: int = 3600,
        default_gpu_seconds: int = 600,
    ):
        self.memory_service = memory_service
        self.default_cpu_seconds = default_cpu_seconds
        self.default_gpu_seconds = default_gpu_seconds
        self._cache: dict[str, SandboxQuota] = {}
        self._logger = logger.bind(service="quota_manager")

    async def get_quota(self, user_id: str) -> SandboxQuota:
        """Get user's current quota, creating if needed."""
        # Check cache
        if user_id in self._cache:
            quota = self._cache[user_id]
            # Check if quota needs reset
            quota = self._maybe_reset_quota(quota)
            return quota

        # Try to load from storage
        quota = await self._load_quota(user_id)
        if not quota:
            quota = SandboxQuota(
                user_id=user_id,
                daily_cpu_seconds=self.default_cpu_seconds,
                daily_gpu_seconds=self.default_gpu_seconds,
            )

        quota = self._maybe_reset_quota(quota)
        self._cache[user_id] = quota
        return quota

    async def check_quota(
        self,
        user_id: str,
        tier: SandboxTier,
        estimated_seconds: int,
    ) -> tuple[bool, str]:
        """
        Check if user has sufficient quota.

        Returns:
            (allowed, reason) tuple
        """
        quota = await self.get_quota(user_id)

        if not quota.can_execute(tier, estimated_seconds):
            from dova.services.sandbox.types import TierConfig

            config = TierConfig.get_config(tier)
            if config.gpu_enabled:
                remaining = quota.daily_gpu_seconds - quota.used_gpu_seconds
                return False, f"GPU quota exceeded. Remaining: {remaining:.0f}s"
            else:
                remaining = quota.daily_cpu_seconds - quota.used_cpu_seconds
                return False, f"CPU quota exceeded. Remaining: {remaining:.0f}s"

        return True, "OK"

    async def reserve_quota(
        self,
        user_id: str,
        tier: SandboxTier,
        seconds: int,
    ) -> bool:
        """
        Reserve quota for an upcoming execution.

        Returns True if reservation succeeded.
        """
        allowed, _ = await self.check_quota(user_id, tier, seconds)
        if not allowed:
            return False

        quota = await self.get_quota(user_id)
        quota.record_usage(tier, seconds)
        self._cache[user_id] = quota

        self._logger.debug(
            "quota_reserved",
            user_id=user_id,
            tier=tier.value,
            seconds=seconds,
        )

        return True

    async def record_usage(
        self,
        user_id: str,
        tier: SandboxTier,
        actual_seconds: float,
        reserved_seconds: int | None = None,
    ) -> None:
        """
        Record actual usage, adjusting for any reservation difference.

        If reserved_seconds is provided and differs from actual_seconds,
        the difference is credited/debited.
        """
        quota = await self.get_quota(user_id)

        if reserved_seconds is not None:
            # Adjust for difference
            adjustment = actual_seconds - reserved_seconds
            quota.record_usage(tier, adjustment)
        else:
            quota.record_usage(tier, actual_seconds)

        self._cache[user_id] = quota
        await self._persist_quota(quota)

        self._logger.info(
            "quota_usage_recorded",
            user_id=user_id,
            tier=tier.value,
            actual_seconds=actual_seconds,
        )

    async def get_remaining(self, user_id: str) -> dict[str, float]:
        """Get user's remaining quota."""
        quota = await self.get_quota(user_id)
        return {
            "cpu_seconds": max(0, quota.daily_cpu_seconds - quota.used_cpu_seconds),
            "gpu_seconds": max(0, quota.daily_gpu_seconds - quota.used_gpu_seconds),
            "reset_in_hours": self._hours_until_reset(quota),
        }

    def _maybe_reset_quota(self, quota: SandboxQuota) -> SandboxQuota:
        """Reset quota if it's past the reset time."""
        now = datetime.utcnow()

        # Check if we're past midnight UTC from the reset date
        next_reset = quota.quota_reset_at.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)

        if now >= next_reset:
            quota.used_cpu_seconds = 0.0
            quota.used_gpu_seconds = 0.0
            quota.quota_reset_at = now
            self._logger.info("quota_reset", user_id=quota.user_id)

        return quota

    def _hours_until_reset(self, quota: SandboxQuota) -> float:
        """Calculate hours until next quota reset."""
        now = datetime.utcnow()
        next_reset = quota.quota_reset_at.replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)
        delta = next_reset - now
        return max(0, delta.total_seconds() / 3600)

    async def _load_quota(self, user_id: str) -> SandboxQuota | None:
        """Load quota from persistent storage."""
        if not self.memory_service:
            return None

        try:
            data = await self.memory_service.get_quota(user_id)
            if data:
                return SandboxQuota(
                    user_id=user_id,
                    daily_cpu_seconds=data.get("daily_cpu_seconds", self.default_cpu_seconds),
                    daily_gpu_seconds=data.get("daily_gpu_seconds", self.default_gpu_seconds),
                    used_cpu_seconds=data.get("used_cpu_seconds", 0.0),
                    used_gpu_seconds=data.get("used_gpu_seconds", 0.0),
                    quota_reset_at=datetime.fromisoformat(data["quota_reset_at"])
                    if data.get("quota_reset_at")
                    else datetime.utcnow(),
                )
        except Exception as e:
            self._logger.warning("quota_load_error", user_id=user_id, error=str(e))

        return None

    async def _persist_quota(self, quota: SandboxQuota) -> None:
        """Persist quota to storage."""
        if not self.memory_service:
            return

        try:
            await self.memory_service.store_quota(
                quota.user_id,
                {
                    "daily_cpu_seconds": quota.daily_cpu_seconds,
                    "daily_gpu_seconds": quota.daily_gpu_seconds,
                    "used_cpu_seconds": quota.used_cpu_seconds,
                    "used_gpu_seconds": quota.used_gpu_seconds,
                    "quota_reset_at": quota.quota_reset_at.isoformat(),
                },
            )
        except Exception as e:
            self._logger.warning("quota_persist_error", user_id=quota.user_id, error=str(e))
