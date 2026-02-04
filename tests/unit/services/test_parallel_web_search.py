"""
Unit tests for ParallelWebSearchService.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from dova.services.web_search import (
    ParallelWebSearchService,
    ParallelSearchConfig,
    ParallelSearchResponse,
    WebSearchResult,
)


@pytest.fixture
def config():
    """Create test configuration with short timeouts."""
    return ParallelSearchConfig(
        per_provider_timeout=5.0,
        overall_timeout=10.0,
        max_results_per_provider=5,
        max_total_results=10,
        max_providers=4,
    )


@pytest.fixture
def service(config):
    """Create ParallelWebSearchService with test config."""
    return ParallelWebSearchService(config=config)


class TestParallelSearchConfig:
    """Tests for ParallelSearchConfig defaults."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ParallelSearchConfig()

        assert config.per_provider_timeout == 15.0
        assert config.overall_timeout == 30.0
        assert config.max_results_per_provider == 10
        assert config.max_total_results == 20
        assert config.max_providers == 4
        assert config.max_depth == 1
        assert config.deduplicate_by_url is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = ParallelSearchConfig(
            per_provider_timeout=5.0,
            max_total_results=50,
        )

        assert config.per_provider_timeout == 5.0
        assert config.max_total_results == 50


class TestParallelWebSearchService:
    """Tests for ParallelWebSearchService."""

    def test_initialization(self, service):
        """Test service initializes correctly."""
        assert service._providers is not None
        assert "duckduckgo" in service._providers
        assert "brave" in service._providers
        assert "tavily" in service._providers
        assert "perplexity" in service._providers

    def test_get_available_providers_default(self, service):
        """Test available providers without API keys."""
        available = service.get_available_providers()

        # Only DuckDuckGo is available without API keys
        assert "duckduckgo" in available
        assert "brave" not in available
        assert "tavily" not in available

    def test_get_available_providers_with_keys(self):
        """Test available providers with API keys configured."""
        service = ParallelWebSearchService(
            brave_api_key="test-brave-key",
            tavily_api_key="test-tavily-key",
        )
        available = service.get_available_providers()

        assert "duckduckgo" in available
        assert "brave" in available
        assert "tavily" in available

    def test_normalize_url(self, service):
        """Test URL normalization."""
        # Basic normalization
        assert service._normalize_url("https://www.example.com/page/") == "example.com/page"
        assert service._normalize_url("https://example.com/page") == "example.com/page"

        # www removal
        assert service._normalize_url("https://www.test.com") == "test.com"

        # Empty URL
        assert service._normalize_url("") == ""

    def test_title_similarity(self, service):
        """Test title similarity calculation."""
        # Identical titles
        assert service._title_similarity("Hello World", "Hello World") == 1.0

        # Similar titles
        similarity = service._title_similarity(
            "AI Research Paper 2025",
            "AI Research Paper 2024"
        )
        assert similarity > 0.5

        # Different titles
        similarity = service._title_similarity(
            "Machine Learning Guide",
            "Cooking Recipes"
        )
        assert similarity < 0.3

        # Empty titles
        assert service._title_similarity("", "Test") == 0.0

    def test_deduplicate_results_by_url(self, service):
        """Test URL-based deduplication."""
        results = [
            WebSearchResult(title="Result 1", url="https://example.com/page", snippet=""),
            WebSearchResult(title="Result 2", url="https://www.example.com/page/", snippet=""),
            WebSearchResult(title="Result 3", url="https://other.com/page", snippet=""),
        ]

        unique, removed = service._deduplicate_results(results)

        assert len(unique) == 2
        assert removed == 1
        assert unique[0].title == "Result 1"
        assert unique[1].title == "Result 3"

    def test_deduplicate_results_by_title(self, service):
        """Test title-based deduplication."""
        results = [
            WebSearchResult(title="AI Research Paper 2025", url="https://a.com", snippet=""),
            WebSearchResult(title="AI Research Paper 2025", url="https://b.com", snippet=""),
            WebSearchResult(title="Different Topic", url="https://c.com", snippet=""),
        ]

        unique, removed = service._deduplicate_results(results)

        assert len(unique) == 2
        assert removed == 1

    def test_rank_results_provider_diversity(self, service):
        """Test result ranking with provider diversity."""
        results = [
            WebSearchResult(title="Result 1", url="https://a.com", snippet="", source_provider="brave", score=0.8),
            WebSearchResult(title="Result 2", url="https://b.com", snippet="", source_provider="brave", score=0.9),
            WebSearchResult(title="Result 3", url="https://c.com", snippet="", source_provider="tavily", score=0.7),
        ]

        ranked = service._rank_results(results)

        # Result from different provider should get diversity boost
        assert len(ranked) == 3
        # The tavily result should be boosted for diversity

    @pytest.mark.asyncio
    async def test_search_with_timeout_success(self, service):
        """Test successful search with timeout."""
        # Mock the provider
        mock_provider = MagicMock()
        mock_provider.name = "test"
        mock_provider.search = AsyncMock(return_value=[
            WebSearchResult(title="Test", url="https://test.com", snippet="")
        ])

        name, results, error = await service._search_with_timeout(
            mock_provider, "test query", 5
        )

        assert name == "test"
        assert len(results) == 1
        assert error is None

    @pytest.mark.asyncio
    async def test_search_with_timeout_timeout(self, service):
        """Test search timeout handling."""
        import asyncio

        async def slow_search(*args):
            await asyncio.sleep(10)
            return []

        mock_provider = MagicMock()
        mock_provider.name = "slow"
        mock_provider.search = slow_search

        # Use very short timeout
        service.config.per_provider_timeout = 0.1

        name, results, error = await service._search_with_timeout(
            mock_provider, "test query", 5
        )

        assert name == "slow"
        assert results == []
        assert "Timeout" in error

    @pytest.mark.asyncio
    async def test_search_with_timeout_exception(self, service):
        """Test search exception handling."""
        mock_provider = MagicMock()
        mock_provider.name = "error"
        mock_provider.search = AsyncMock(side_effect=Exception("Test error"))

        name, results, error = await service._search_with_timeout(
            mock_provider, "test query", 5
        )

        assert name == "error"
        assert results == []
        assert "Test error" in error

    @pytest.mark.asyncio
    async def test_search_parallel_no_providers(self, service):
        """Test parallel search with no available providers."""
        # Force no providers available
        for provider in service._providers.values():
            provider.is_available = MagicMock(return_value=False)

        response = await service.search_parallel("test query")

        assert response.error is not None
        assert "No search providers" in response.error

    @pytest.mark.asyncio
    async def test_search_parallel_success(self, service):
        """Test successful parallel search."""
        # Mock DuckDuckGo (always available)
        mock_results = [
            WebSearchResult(title="DDG Result 1", url="https://ddg1.com", snippet="", source_provider="duckduckgo"),
            WebSearchResult(title="DDG Result 2", url="https://ddg2.com", snippet="", source_provider="duckduckgo"),
        ]

        with patch.object(
            service._providers["duckduckgo"],
            "search",
            new=AsyncMock(return_value=mock_results)
        ):
            response = await service.search_parallel("test query")

        assert response.error is None
        assert len(response.results) == 2
        assert "duckduckgo" in response.providers_succeeded
        assert response.total_time_ms > 0

    @pytest.mark.asyncio
    async def test_search_parallel_multiple_providers(self):
        """Test parallel search with multiple providers."""
        service = ParallelWebSearchService(
            brave_api_key="test-key",
            config=ParallelSearchConfig(
                per_provider_timeout=5.0,
                overall_timeout=10.0,
            ),
        )

        # Mock both providers
        brave_results = [
            WebSearchResult(title="Brave Result", url="https://brave.com", snippet="", source_provider="brave"),
        ]
        ddg_results = [
            WebSearchResult(title="DDG Result", url="https://ddg.com", snippet="", source_provider="duckduckgo"),
        ]

        with patch.object(
            service._providers["brave"],
            "search",
            new=AsyncMock(return_value=brave_results)
        ):
            with patch.object(
                service._providers["duckduckgo"],
                "search",
                new=AsyncMock(return_value=ddg_results)
            ):
                response = await service.search_parallel("test query")

        assert response.error is None
        assert len(response.results) == 2
        assert "brave" in response.providers_succeeded
        assert "duckduckgo" in response.providers_succeeded

    @pytest.mark.asyncio
    async def test_search_parallel_partial_failure(self):
        """Test parallel search with some providers failing."""
        service = ParallelWebSearchService(
            brave_api_key="test-key",
            config=ParallelSearchConfig(per_provider_timeout=5.0),
        )

        # Mock brave to fail, ddg to succeed
        ddg_results = [
            WebSearchResult(title="DDG Result", url="https://ddg.com", snippet="", source_provider="duckduckgo"),
        ]

        with patch.object(
            service._providers["brave"],
            "search",
            new=AsyncMock(side_effect=Exception("Brave error"))
        ):
            with patch.object(
                service._providers["duckduckgo"],
                "search",
                new=AsyncMock(return_value=ddg_results)
            ):
                response = await service.search_parallel("test query")

        assert response.error is None  # Partial success, not an error
        assert len(response.results) == 1
        assert "duckduckgo" in response.providers_succeeded
        assert "brave" in response.providers_failed

    @pytest.mark.asyncio
    async def test_search_parallel_respects_max_providers(self):
        """Test that max_providers limit is respected."""
        service = ParallelWebSearchService(
            brave_api_key="test-key",
            tavily_api_key="test-key",
            perplexity_api_key="test-key",
            config=ParallelSearchConfig(max_providers=2),
        )

        # All providers should be available
        available = service.get_available_providers()
        assert len(available) == 4

        # But only max_providers should be searched
        with patch.object(
            service._providers["brave"],
            "search",
            new=AsyncMock(return_value=[])
        ):
            with patch.object(
                service._providers["perplexity"],
                "search",
                new=AsyncMock(return_value=[])
            ):
                with patch.object(
                    service._providers["tavily"],
                    "search",
                    new=AsyncMock(return_value=[])
                ):
                    with patch.object(
                        service._providers["duckduckgo"],
                        "search",
                        new=AsyncMock(return_value=[])
                    ):
                        response = await service.search_parallel("test query")

        assert len(response.providers_searched) == 2

    @pytest.mark.asyncio
    async def test_search_parallel_respects_max_results(self, service):
        """Test that max_total_results is respected."""
        # Create more results than max
        mock_results = [
            WebSearchResult(title=f"Result {i}", url=f"https://test{i}.com", snippet="", source_provider="duckduckgo")
            for i in range(20)
        ]

        service.config.max_total_results = 5

        with patch.object(
            service._providers["duckduckgo"],
            "search",
            new=AsyncMock(return_value=mock_results)
        ):
            response = await service.search_parallel("test query")

        assert len(response.results) == 5


class TestParallelSearchResponse:
    """Tests for ParallelSearchResponse."""

    def test_default_values(self):
        """Test default response values."""
        response = ParallelSearchResponse()

        assert response.results == []
        assert response.providers_searched == []
        assert response.providers_succeeded == []
        assert response.providers_failed == {}
        assert response.query == ""
        assert response.total_time_ms == 0.0
        assert response.deduplicated_count == 0
        assert response.error is None
