"""Tests for memory strategies module."""


from dova.services.memory_strategies import (
    MemoryStrategy,
    MemoryStrategyConfig,
    NamespacedRetrievalConfig,
    create_default_strategies,
)


class TestMemoryStrategy:
    """Tests for MemoryStrategy enum."""

    def test_strategy_values(self):
        """Test all strategy enum values exist."""
        assert MemoryStrategy.SUMMARY.value == "summary"
        assert MemoryStrategy.USER_PREFERENCE.value == "user_preference"
        assert MemoryStrategy.SEMANTIC.value == "semantic"


class TestNamespacedRetrievalConfig:
    """Tests for NamespacedRetrievalConfig dataclass."""

    def test_creation_with_defaults(self):
        """Test creating config with default values."""
        config = NamespacedRetrievalConfig(namespace="/summaries/{actorId}")

        assert config.namespace == "/summaries/{actorId}"
        assert config.top_k == 5
        assert config.relevance_score == 0.7

    def test_creation_with_custom_values(self):
        """Test creating config with custom values."""
        config = NamespacedRetrievalConfig(
            namespace="/facts/{actorId}",
            top_k=20,
            relevance_score=0.9,
        )

        assert config.namespace == "/facts/{actorId}"
        assert config.top_k == 20
        assert config.relevance_score == 0.9


class TestMemoryStrategyConfig:
    """Tests for MemoryStrategyConfig dataclass."""

    def test_creation_empty(self):
        """Test creating empty config."""
        config = MemoryStrategyConfig()

        assert config.strategies == []
        assert config.session_id == ""
        assert config.actor_id == ""
        assert config.memory_id == ""

    def test_to_retrieval_config(self):
        """Test converting to retrieval config format."""
        strategies = [
            NamespacedRetrievalConfig(namespace="/summaries/{actorId}", top_k=5),
            NamespacedRetrievalConfig(
                namespace="/facts/{actorId}", top_k=10, relevance_score=0.8
            ),
        ]

        config = MemoryStrategyConfig(
            strategies=strategies,
            actor_id="user-123",
        )

        retrieval_config = config.to_retrieval_config()

        assert "/summaries/user-123" in retrieval_config
        assert retrieval_config["/summaries/user-123"]["top_k"] == 5

        assert "/facts/user-123" in retrieval_config
        assert retrieval_config["/facts/user-123"]["top_k"] == 10
        assert retrieval_config["/facts/user-123"]["relevance_score"] == 0.8


class TestCreateDefaultStrategies:
    """Tests for create_default_strategies function."""

    def test_all_strategies_enabled(self):
        """Test creating strategies with all enabled."""
        strategies = create_default_strategies(
            actor_id="user-123",
            summary_enabled=True,
            preference_enabled=True,
            semantic_enabled=True,
        )

        assert len(strategies) == 3

        namespaces = [s.namespace for s in strategies]
        assert "/summaries/{actorId}" in namespaces
        assert "/preferences/{actorId}" in namespaces
        assert "/facts/{actorId}" in namespaces

    def test_only_summary_enabled(self):
        """Test creating strategies with only summary."""
        strategies = create_default_strategies(
            actor_id="user-123",
            summary_enabled=True,
            preference_enabled=False,
            semantic_enabled=False,
        )

        assert len(strategies) == 1
        assert strategies[0].namespace == "/summaries/{actorId}"

    def test_no_strategies_enabled(self):
        """Test creating strategies with none enabled."""
        strategies = create_default_strategies(
            actor_id="user-123",
            summary_enabled=False,
            preference_enabled=False,
            semantic_enabled=False,
        )

        assert len(strategies) == 0

    def test_custom_top_k_values(self):
        """Test custom top_k values are applied."""
        strategies = create_default_strategies(
            actor_id="user-123",
            summary_top_k=10,
            preference_top_k=15,
            semantic_top_k=25,
            semantic_relevance=0.85,
        )

        summary_config = next(
            s for s in strategies if "summaries" in s.namespace
        )
        preference_config = next(
            s for s in strategies if "preferences" in s.namespace
        )
        semantic_config = next(
            s for s in strategies if "facts" in s.namespace
        )

        assert summary_config.top_k == 10
        assert preference_config.top_k == 15
        assert semantic_config.top_k == 25
        assert semantic_config.relevance_score == 0.85
