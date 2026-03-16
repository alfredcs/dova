"""
Multi-provider Web Search Service for DOVA.

Supports multiple web search providers with auto-selection and fallback:
- DuckDuckGo (free, no API key required - always available fallback)
- Brave Search (structured results, free tier available)
- Perplexity Sonar (AI-synthesized answers with citations)
- Tavily (existing provider)

Features:
- Parallel search across multiple providers
- Result aggregation and deduplication
- Safety guards (timeouts, depth limits, max results)
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse

import httpx
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class WebSearchResult:
    """Unified web search result."""

    title: str
    url: str
    snippet: str
    published_date: str | None = None
    source_provider: str = ""
    score: float = 0.0


@dataclass
class WebSearchResponse:
    """Response from web search."""

    results: list[WebSearchResult] = field(default_factory=list)
    provider: str = ""
    query: str = ""
    error: str | None = None


class WebSearchProvider(ABC):
    """Abstract base class for web search providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and available."""
        pass

    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> list[WebSearchResult]:
        """Execute search and return results."""
        pass


class DuckDuckGoProvider(WebSearchProvider):
    """DuckDuckGo search provider - free, no API key required."""

    @property
    def name(self) -> str:
        return "duckduckgo"

    def is_available(self) -> bool:
        return True  # Always available

    async def search(self, query: str, max_results: int = 10) -> list[WebSearchResult]:
        """Search using DuckDuckGo (runs sync client in executor to avoid blocking)."""
        try:
            from ddgs import DDGS

            def _sync_search() -> list[dict]:
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=max_results))

            loop = asyncio.get_event_loop()
            raw_results = await loop.run_in_executor(None, _sync_search)

            return [
                WebSearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", r.get("link", "")),
                    snippet=r.get("body", r.get("snippet", "")),
                    published_date=None,
                    source_provider=self.name,
                )
                for r in raw_results
            ]
        except Exception as e:
            logger.warning("duckduckgo_search_error", error=str(e))
            return []


class BraveSearchProvider(WebSearchProvider):
    """Brave Search provider - structured results with optional API key."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self._base_url = "https://api.search.brave.com/res/v1/web/search"
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "brave"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str, max_results: int = 10) -> list[WebSearchResult]:
        """Search using Brave Search API."""
        if not self.api_key:
            return []

        try:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=30.0)

            response = await self._client.get(
                self._base_url,
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": self.api_key,
                },
                params={
                    "q": query,
                    "count": max_results,
                    "freshness": "pw",  # Past week for recent results
                },
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("web", {}).get("results", []):
                results.append(
                    WebSearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("description", ""),
                        published_date=item.get("age"),
                        source_provider=self.name,
                    )
                )
            return results

        except Exception as e:
            logger.warning("brave_search_error", error=str(e))
            return []


class PerplexityProvider(WebSearchProvider):
    """Perplexity Sonar provider - AI-synthesized answers with citations."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self._base_url = "https://api.perplexity.ai/chat/completions"
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "perplexity"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str, max_results: int = 10) -> list[WebSearchResult]:
        """Search using Perplexity Sonar API."""
        if not self.api_key:
            return []

        try:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=30.0)

            response = await self._client.post(
                self._base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "sonar",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a search assistant. Provide factual information with sources.",
                        },
                        {"role": "user", "content": query},
                    ],
                    "return_citations": True,
                },
            )
            response.raise_for_status()
            data = response.json()

            results = []
            # Perplexity returns a synthesized answer with citations
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            citations = data.get("citations", [])

            # Add the synthesized answer as the first result
            if content:
                results.append(
                    WebSearchResult(
                        title="Perplexity AI Summary",
                        url="",
                        snippet=content[:500],
                        source_provider=self.name,
                        score=1.0,
                    )
                )

            # Add citations as additional results
            for citation in citations[:max_results - 1]:
                if isinstance(citation, str):
                    results.append(
                        WebSearchResult(
                            title="Source",
                            url=citation,
                            snippet="",
                            source_provider=self.name,
                        )
                    )
                elif isinstance(citation, dict):
                    results.append(
                        WebSearchResult(
                            title=citation.get("title", "Source"),
                            url=citation.get("url", ""),
                            snippet=citation.get("snippet", ""),
                            source_provider=self.name,
                        )
                    )

            return results

        except Exception as e:
            logger.warning("perplexity_search_error", error=str(e))
            return []


class TavilyProvider(WebSearchProvider):
    """Tavily search provider - existing implementation wrapper."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self._client = None

    @property
    def name(self) -> str:
        return "tavily"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str, max_results: int = 10) -> list[WebSearchResult]:
        """Search using Tavily API (runs sync client in executor to avoid blocking)."""
        if not self.api_key:
            return []

        try:
            from tavily import TavilyClient

            if self._client is None:
                self._client = TavilyClient(api_key=self.api_key)

            client = self._client

            def _sync_search() -> dict:
                return client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=max_results,
                )

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, _sync_search)

            results = []
            for item in response.get("results", []):
                if not isinstance(item, dict):
                    continue
                results.append(
                    WebSearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("content", "")[:500],
                        published_date=item.get("published_date"),
                        source_provider=self.name,
                        score=item.get("score", 0),
                    )
                )
            return results

        except ImportError:
            logger.warning("tavily_not_installed")
            return []
        except Exception as e:
            logger.warning("tavily_search_error", error=str(e))
            return []


ProviderType = Literal["auto", "brave", "perplexity", "tavily", "duckduckgo"]


class WebSearchService:
    """
    Multi-provider web search service with auto-selection and fallback.

    Provider priority (when auto):
    1. Brave Search (when BRAVE_API_KEY is set)
    2. Perplexity (when PERPLEXITY_API_KEY is set)
    3. Tavily (when TAVILY_API_KEY is set)
    4. DuckDuckGo (always available as fallback)
    """

    def __init__(
        self,
        provider: ProviderType = "auto",
        brave_api_key: str | None = None,
        perplexity_api_key: str | None = None,
        tavily_api_key: str | None = None,
        fallback_enabled: bool = True,
    ):
        self.preferred_provider = provider
        self.fallback_enabled = fallback_enabled

        # Initialize providers
        self._providers: dict[str, WebSearchProvider] = {
            "brave": BraveSearchProvider(api_key=brave_api_key),
            "perplexity": PerplexityProvider(api_key=perplexity_api_key),
            "tavily": TavilyProvider(api_key=tavily_api_key),
            "duckduckgo": DuckDuckGoProvider(),
        }

        # Provider priority order for auto-selection
        self._priority_order = ["brave", "perplexity", "tavily", "duckduckgo"]

    def _get_provider(self) -> WebSearchProvider:
        """Select provider based on configuration and availability."""
        if self.preferred_provider != "auto":
            provider = self._providers.get(self.preferred_provider)
            if provider and provider.is_available():
                return provider
            elif self.fallback_enabled:
                logger.info(
                    "preferred_provider_unavailable",
                    provider=self.preferred_provider,
                    fallback="auto",
                )
            else:
                raise ValueError(f"Provider {self.preferred_provider} is not available")

        # Auto-select based on priority
        for name in self._priority_order:
            provider = self._providers.get(name)
            if provider and provider.is_available():
                logger.debug("auto_selected_provider", provider=name)
                return provider

        # Should never reach here since DuckDuckGo is always available
        return self._providers["duckduckgo"]

    def get_available_providers(self) -> list[str]:
        """Get list of available (configured) providers."""
        return [name for name, p in self._providers.items() if p.is_available()]

    async def search(
        self,
        query: str,
        max_results: int = 10,
        provider: ProviderType | None = None,
    ) -> WebSearchResponse:
        """
        Execute web search with automatic provider selection and fallback.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            provider: Override provider selection (optional)

        Returns:
            WebSearchResponse with results and metadata
        """
        # Determine which provider to use
        if provider and provider != "auto":
            selected_provider = self._providers.get(provider)
            if not selected_provider or not selected_provider.is_available():
                if self.fallback_enabled:
                    selected_provider = self._get_provider()
                else:
                    return WebSearchResponse(
                        query=query,
                        error=f"Provider {provider} is not available",
                    )
        else:
            selected_provider = self._get_provider()

        logger.info(
            "web_search_starting",
            query=query,
            provider=selected_provider.name,
            max_results=max_results,
        )

        # Try primary provider
        try:
            results = await selected_provider.search(query, max_results)
            if results:
                return WebSearchResponse(
                    results=results,
                    provider=selected_provider.name,
                    query=query,
                )
        except Exception as e:
            logger.warning(
                "primary_provider_failed",
                provider=selected_provider.name,
                error=str(e),
            )

        # Fallback to other providers if enabled
        if self.fallback_enabled:
            for name in self._priority_order:
                if name == selected_provider.name:
                    continue
                fallback = self._providers.get(name)
                if fallback and fallback.is_available():
                    try:
                        logger.info("trying_fallback_provider", provider=name)
                        results = await fallback.search(query, max_results)
                        if results:
                            return WebSearchResponse(
                                results=results,
                                provider=name,
                                query=query,
                            )
                    except Exception as e:
                        logger.warning(
                            "fallback_provider_failed",
                            provider=name,
                            error=str(e),
                        )
                        continue

        return WebSearchResponse(
            query=query,
            error="All search providers failed",
        )


@dataclass
class ParallelSearchConfig:
    """Configuration for parallel web search with safety guards."""

    # Timeout settings
    per_provider_timeout: float = 15.0  # Timeout per provider in seconds
    overall_timeout: float = 30.0  # Overall timeout for all searches

    # Result limits
    max_results_per_provider: int = 10  # Max results from each provider
    max_total_results: int = 20  # Max results after aggregation
    max_providers: int = 4  # Max providers to search in parallel

    # Depth/recursion limits (for future deep search)
    max_depth: int = 1  # Max recursion depth (1 = single search, no follow-up)
    max_follow_up_queries: int = 0  # Max follow-up queries per search

    # Deduplication
    deduplicate_by_url: bool = True
    deduplicate_by_title_similarity: float = 0.9  # Title similarity threshold

    # Ranking weights
    provider_diversity_weight: float = 0.3  # Boost for diverse sources
    recency_weight: float = 0.2  # Boost for recent results
    relevance_weight: float = 0.5  # Base relevance weight


@dataclass
class ParallelSearchResponse:
    """Response from parallel web search."""

    results: list[WebSearchResult] = field(default_factory=list)
    providers_searched: list[str] = field(default_factory=list)
    providers_succeeded: list[str] = field(default_factory=list)
    providers_failed: dict[str, str] = field(default_factory=dict)
    query: str = ""
    total_time_ms: float = 0.0
    deduplicated_count: int = 0
    error: str | None = None


class ParallelWebSearchService:
    """
    Parallel web search service that aggregates results from multiple providers.

    Features:
    - Concurrent searches across all available providers
    - Per-provider and overall timeouts
    - Result deduplication by URL and title similarity
    - Ranking by relevance, recency, and source diversity
    - Configurable safety guards

    Example:
        service = ParallelWebSearchService(
            brave_api_key="...",
            tavily_api_key="...",
        )
        response = await service.search_parallel("SpaceX xAI merger")
        print(f"Found {len(response.results)} unique results from {response.providers_succeeded}")
    """

    def __init__(
        self,
        brave_api_key: str | None = None,
        perplexity_api_key: str | None = None,
        tavily_api_key: str | None = None,
        config: ParallelSearchConfig | None = None,
    ):
        self.config = config or ParallelSearchConfig()

        # Initialize providers
        self._providers: dict[str, WebSearchProvider] = {
            "brave": BraveSearchProvider(api_key=brave_api_key),
            "perplexity": PerplexityProvider(api_key=perplexity_api_key),
            "tavily": TavilyProvider(api_key=tavily_api_key),
            "duckduckgo": DuckDuckGoProvider(),
        }

        # Provider priority for ranking ties
        self._priority = {"brave": 1, "tavily": 2, "perplexity": 3, "duckduckgo": 4}

    def get_available_providers(self) -> list[str]:
        """Get list of available (configured) providers."""
        return [name for name, p in self._providers.items() if p.is_available()]

    async def _search_with_timeout(
        self,
        provider: WebSearchProvider,
        query: str,
        max_results: int,
    ) -> tuple[str, list[WebSearchResult], str | None]:
        """
        Execute search with per-provider timeout.

        Returns:
            Tuple of (provider_name, results, error_message)
        """
        try:
            results = await asyncio.wait_for(
                provider.search(query, max_results),
                timeout=self.config.per_provider_timeout,
            )
            return (provider.name, results, None)
        except asyncio.TimeoutError:
            return (provider.name, [], f"Timeout after {self.config.per_provider_timeout}s")
        except Exception as e:
            return (provider.name, [], str(e))

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        if not url:
            return ""
        try:
            parsed = urlparse(url)
            # Remove www prefix and trailing slashes
            domain = parsed.netloc.lower().replace("www.", "")
            path = parsed.path.rstrip("/")
            return f"{domain}{path}"
        except Exception:
            return url.lower()

    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate simple title similarity (Jaccard)."""
        if not title1 or not title2:
            return 0.0
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _deduplicate_results(
        self,
        all_results: list[WebSearchResult],
    ) -> tuple[list[WebSearchResult], int]:
        """
        Deduplicate results by URL and title similarity.

        Returns:
            Tuple of (deduplicated_results, removed_count)
        """
        if not self.config.deduplicate_by_url:
            return all_results, 0

        seen_urls: set[str] = set()
        seen_titles: list[str] = []
        unique_results: list[WebSearchResult] = []
        removed_count = 0

        for result in all_results:
            # Check URL deduplication
            normalized_url = self._normalize_url(result.url)
            if normalized_url and normalized_url in seen_urls:
                removed_count += 1
                continue

            # Check title similarity
            is_duplicate = False
            for seen_title in seen_titles:
                if self._title_similarity(result.title, seen_title) >= self.config.deduplicate_by_title_similarity:
                    is_duplicate = True
                    removed_count += 1
                    break

            if not is_duplicate:
                seen_urls.add(normalized_url)
                seen_titles.append(result.title)
                unique_results.append(result)

        return unique_results, removed_count

    def _rank_results(
        self,
        results: list[WebSearchResult],
    ) -> list[WebSearchResult]:
        """
        Rank results by relevance, recency, and source diversity.
        """
        if not results:
            return results

        # Track provider counts for diversity scoring
        provider_counts: dict[str, int] = {}

        def score_result(result: WebSearchResult) -> float:
            """Calculate composite score for a result."""
            score = 0.0

            # Base relevance score
            score += result.score * self.config.relevance_weight

            # Recency bonus (if date available)
            if result.published_date:
                score += 0.1 * self.config.recency_weight

            # Provider diversity bonus
            provider = result.source_provider
            count = provider_counts.get(provider, 0)
            provider_counts[provider] = count + 1

            # First result from a provider gets full diversity bonus
            # Subsequent results get diminishing bonus
            diversity_bonus = self.config.provider_diversity_weight / (count + 1)
            score += diversity_bonus

            # Provider priority tiebreaker
            priority = self._priority.get(provider, 5)
            score += (5 - priority) * 0.01

            return score

        # Score and sort results
        scored = [(r, score_result(r)) for r in results]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [r for r, _ in scored]

    async def search_parallel(
        self,
        query: str,
        providers: list[str] | None = None,
        max_results: int | None = None,
    ) -> ParallelSearchResponse:
        """
        Execute parallel search across multiple providers with aggregation.

        Args:
            query: Search query string
            providers: List of providers to use (None = all available)
            max_results: Max results to return (None = use config default)

        Returns:
            ParallelSearchResponse with aggregated, deduplicated, ranked results
        """
        import time
        start_time = time.time()

        max_results = max_results or self.config.max_total_results
        max_per_provider = self.config.max_results_per_provider

        # Determine which providers to use
        if providers:
            providers_to_use = [
                name for name in providers
                if name in self._providers and self._providers[name].is_available()
            ]
        else:
            providers_to_use = self.get_available_providers()

        # Apply max_providers limit
        providers_to_use = providers_to_use[:self.config.max_providers]

        if not providers_to_use:
            return ParallelSearchResponse(
                query=query,
                error="No search providers available",
            )

        logger.info(
            "parallel_search_starting",
            query=query,
            providers=providers_to_use,
            max_per_provider=max_per_provider,
        )

        # Create search tasks for all providers
        tasks = [
            self._search_with_timeout(
                self._providers[name],
                query,
                max_per_provider,
            )
            for name in providers_to_use
        ]

        # Execute all searches in parallel with overall timeout
        try:
            results_tuples = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.overall_timeout,
            )
        except asyncio.TimeoutError:
            total_time = (time.time() - start_time) * 1000
            return ParallelSearchResponse(
                query=query,
                providers_searched=providers_to_use,
                total_time_ms=total_time,
                error=f"Overall timeout after {self.config.overall_timeout}s",
            )

        # Process results from each provider
        all_results: list[WebSearchResult] = []
        providers_succeeded: list[str] = []
        providers_failed: dict[str, str] = {}

        for result in results_tuples:
            if isinstance(result, Exception):
                # This shouldn't happen since we catch exceptions in _search_with_timeout
                continue

            provider_name, provider_results, error = result

            if error:
                providers_failed[provider_name] = error
                logger.warning(
                    "parallel_search_provider_failed",
                    provider=provider_name,
                    error=error,
                )
            else:
                providers_succeeded.append(provider_name)
                all_results.extend(provider_results)
                logger.info(
                    "parallel_search_provider_succeeded",
                    provider=provider_name,
                    result_count=len(provider_results),
                )

        # Deduplicate results
        unique_results, dedup_count = self._deduplicate_results(all_results)

        # Rank results
        ranked_results = self._rank_results(unique_results)

        # Apply max results limit
        final_results = ranked_results[:max_results]

        total_time = (time.time() - start_time) * 1000

        logger.info(
            "parallel_search_complete",
            query=query,
            providers_succeeded=providers_succeeded,
            providers_failed=list(providers_failed.keys()),
            total_results=len(all_results),
            unique_results=len(unique_results),
            final_results=len(final_results),
            deduplicated_count=dedup_count,
            total_time_ms=total_time,
        )

        return ParallelSearchResponse(
            results=final_results,
            providers_searched=providers_to_use,
            providers_succeeded=providers_succeeded,
            providers_failed=providers_failed,
            query=query,
            total_time_ms=total_time,
            deduplicated_count=dedup_count,
        )


def create_web_search_service() -> WebSearchService:
    """Create WebSearchService from environment/settings."""
    import os

    return WebSearchService(
        provider="auto",
        brave_api_key=os.environ.get("BRAVE_API_KEY"),
        perplexity_api_key=os.environ.get("PERPLEXITY_API_KEY"),
        tavily_api_key=os.environ.get("TAVILY_API_KEY") or os.environ.get("MCP_TAVILY_API_KEY"),
        fallback_enabled=True,
    )


def create_parallel_search_service(
    config: ParallelSearchConfig | None = None,
) -> ParallelWebSearchService:
    """Create ParallelWebSearchService from environment/settings."""
    import os

    return ParallelWebSearchService(
        brave_api_key=os.environ.get("BRAVE_API_KEY"),
        perplexity_api_key=os.environ.get("PERPLEXITY_API_KEY"),
        tavily_api_key=os.environ.get("TAVILY_API_KEY") or os.environ.get("MCP_TAVILY_API_KEY"),
        config=config,
    )
