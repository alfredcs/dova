"""
Multi-tiered thinking level system for LLM reasoning.

Provides configurable thinking budgets to balance response quality
with token usage and latency.
"""

from dataclasses import dataclass
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class ThinkingLevel(Enum):
    """Thinking levels with associated token budgets."""

    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


# Token budgets for each level
THINKING_BUDGETS: dict[ThinkingLevel, int] = {
    ThinkingLevel.OFF: 0,
    ThinkingLevel.MINIMAL: 1024,
    ThinkingLevel.LOW: 4096,
    ThinkingLevel.MEDIUM: 16384,
    ThinkingLevel.HIGH: 32768,
    ThinkingLevel.XHIGH: 65536,
}


@dataclass
class ThinkingConfig:
    """Configuration for thinking/reasoning budget."""

    level: ThinkingLevel
    budget_tokens: int
    auto_select: bool = True

    @classmethod
    def from_level(cls, level: ThinkingLevel, auto_select: bool = True) -> "ThinkingConfig":
        """Create config from a thinking level."""
        return cls(
            level=level,
            budget_tokens=THINKING_BUDGETS[level],
            auto_select=auto_select,
        )

    @classmethod
    def default(cls) -> "ThinkingConfig":
        """Create default thinking config (medium level)."""
        return cls.from_level(ThinkingLevel.MEDIUM)


class ThinkingService:
    """Service for managing thinking levels and budgets."""

    def __init__(self, default_level: ThinkingLevel = ThinkingLevel.MEDIUM):
        self.default_level = default_level
        self._logger = logger.bind(service="thinking")

    def get_budget_for_level(self, level: ThinkingLevel) -> int:
        """Get token budget for a thinking level."""
        return THINKING_BUDGETS[level]

    def select_level_for_task(
        self,
        task_type: str,
        query: str | None = None,
        complexity_hint: str | None = None,
    ) -> ThinkingLevel:
        """
        Auto-select thinking level based on task characteristics.

        Args:
            task_type: Type of task (e.g., "reasoning", "summarization")
            query: Optional query text for complexity estimation
            complexity_hint: Optional hint ("simple", "moderate", "complex")

        Returns:
            Appropriate thinking level for the task
        """
        # Task-type based defaults
        task_levels: dict[str, ThinkingLevel] = {
            "embedding": ThinkingLevel.OFF,
            "classification": ThinkingLevel.MINIMAL,
            "summarization": ThinkingLevel.LOW,
            "code_generation": ThinkingLevel.MEDIUM,
            "reasoning": ThinkingLevel.HIGH,
            "research": ThinkingLevel.HIGH,
            "analysis": ThinkingLevel.HIGH,
        }

        base_level = task_levels.get(task_type.lower(), self.default_level)

        # Adjust based on complexity hint
        if complexity_hint:
            hint_adjustments: dict[str, int] = {
                "simple": -1,
                "moderate": 0,
                "complex": 1,
                "very_complex": 2,
            }
            adjustment = hint_adjustments.get(complexity_hint.lower(), 0)
            base_level = self._adjust_level(base_level, adjustment)

        # Adjust based on query length (rough complexity proxy)
        if query:
            query_len = len(query)
            if query_len > 2000:
                base_level = self._adjust_level(base_level, 1)
            elif query_len < 50:
                base_level = self._adjust_level(base_level, -1)

        self._logger.debug(
            "thinking_level_selected",
            task_type=task_type,
            level=base_level.value,
            budget=THINKING_BUDGETS[base_level],
        )

        return base_level

    def create_thinking_params(self, level: ThinkingLevel) -> dict:
        """
        Create thinking parameters for LLM request.

        Args:
            level: Thinking level to use

        Returns:
            Dict with thinking configuration for LLM API
        """
        if level == ThinkingLevel.OFF:
            return {}

        budget = THINKING_BUDGETS[level]
        return {
            "thinking": {
                "type": "enabled",
                "budget_tokens": budget,
            }
        }

    def create_config(
        self,
        level: ThinkingLevel | None = None,
        auto_select: bool = True,
    ) -> ThinkingConfig:
        """Create a thinking configuration."""
        effective_level = level or self.default_level
        return ThinkingConfig(
            level=effective_level,
            budget_tokens=THINKING_BUDGETS[effective_level],
            auto_select=auto_select,
        )

    def _adjust_level(self, level: ThinkingLevel, adjustment: int) -> ThinkingLevel:
        """Adjust thinking level by step count."""
        levels = list(ThinkingLevel)
        current_idx = levels.index(level)
        new_idx = max(0, min(len(levels) - 1, current_idx + adjustment))
        return levels[new_idx]
