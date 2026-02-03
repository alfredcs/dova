"""Memory strategy configurations for AgentCore Memory.

Defines memory strategy types and namespaced retrieval configurations
for integrating with AWS Bedrock AgentCore Memory service.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryStrategy(Enum):
    """Available memory strategies for AgentCore Memory."""

    SUMMARY = "summary"
    USER_PREFERENCE = "user_preference"
    SEMANTIC = "semantic"


@dataclass
class NamespacedRetrievalConfig:
    """Configuration for namespaced memory retrieval.

    AgentCore Memory uses namespaces to organize different types of memory:
    - /summaries/{actorId} - Conversation summaries
    - /preferences/{actorId} - User preferences
    - /facts/{actorId} - Semantic facts/knowledge

    Args:
        namespace: Memory namespace pattern (e.g., "/preferences/{actorId}")
        top_k: Number of memories to retrieve
        relevance_score: Minimum relevance score (0.0-1.0) for semantic retrieval
    """

    namespace: str
    top_k: int = 5
    relevance_score: float = 0.7


@dataclass
class MemoryStrategyConfig:
    """Configuration for a complete memory strategy setup.

    Combines multiple retrieval configurations for different memory types.
    """

    strategies: list[NamespacedRetrievalConfig] = field(default_factory=list)
    session_id: str = ""
    actor_id: str = ""
    memory_id: str = ""

    def to_retrieval_config(self) -> dict[str, Any]:
        """Convert to AgentCore retrieval config format.

        Returns:
            Dictionary mapping namespaces to retrieval configs
        """
        config = {}
        for strategy in self.strategies:
            # Substitute actor_id in namespace
            namespace = strategy.namespace.format(actorId=self.actor_id)
            config[namespace] = {
                "top_k": strategy.top_k,
                "relevance_score": strategy.relevance_score,
            }
        return config


def create_default_strategies(
    actor_id: str,  # noqa: ARG001 - Used in namespace patterns
    summary_enabled: bool = True,
    preference_enabled: bool = True,
    semantic_enabled: bool = True,
    summary_top_k: int = 5,
    preference_top_k: int = 5,
    semantic_top_k: int = 10,
    semantic_relevance: float = 0.7,
) -> list[NamespacedRetrievalConfig]:
    """Create default memory strategy configurations.

    Args:
        actor_id: User/actor identifier
        summary_enabled: Enable summary retrieval
        preference_enabled: Enable user preference retrieval
        semantic_enabled: Enable semantic memory retrieval
        summary_top_k: Number of summaries to retrieve
        preference_top_k: Number of preferences to retrieve
        semantic_top_k: Number of semantic memories to retrieve
        semantic_relevance: Minimum relevance score for semantic retrieval

    Returns:
        List of configured retrieval strategies
    """
    strategies = []

    if summary_enabled:
        strategies.append(
            NamespacedRetrievalConfig(
                namespace="/summaries/{actorId}",
                top_k=summary_top_k,
            )
        )

    if preference_enabled:
        strategies.append(
            NamespacedRetrievalConfig(
                namespace="/preferences/{actorId}",
                top_k=preference_top_k,
            )
        )

    if semantic_enabled:
        strategies.append(
            NamespacedRetrievalConfig(
                namespace="/facts/{actorId}",
                top_k=semantic_top_k,
                relevance_score=semantic_relevance,
            )
        )

    return strategies
