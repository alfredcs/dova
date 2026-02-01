"""
Multi-provider Web Search Service for DOVA.

Supports multiple web search providers with auto-selection and fallback:
- DuckDuckGo (free, no API key required - always available fallback)
- Brave Search (structured results, free tier available)
- Perplexity Sonar (AI-synthesized answers with citations)
- Tavily (existing provider)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

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
        """Search using DuckDuckGo."""
        try:
            from ddgs import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(
                        WebSearchResult(
                            title=r.get("title", ""),
                            url=r.get("href", r.get("link", "")),
                            snippet=r.get("body", r.get("snippet", "")),
                            published_date=None,  # DDG doesn't provide dates
                            source_provider=self.name,
                        )
                    )
            return results
        except Exception as e:
            logger.warning("duckduckgo_search_error", error=str(e))
            return []


class BraveSearchProvider(WebSearchProvider):
    """Brave Search provider - structured results with optional API key."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self._base_url = "https://api.search.brave.com/res/v1/web/search"

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
            async with httpx.AsyncClient() as client:
                response = await client.get(
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
                    timeout=30.0,
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
            async with httpx.AsyncClient() as client:
                response = await client.post(
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
                    timeout=30.0,
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
        """Search using Tavily API."""
        if not self.api_key:
            return []

        try:
            from tavily import TavilyClient

            if self._client is None:
                self._client = TavilyClient(api_key=self.api_key)

            response = self._client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
            )

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
