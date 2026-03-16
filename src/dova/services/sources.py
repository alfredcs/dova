"""Source registry and fetcher service."""
import hashlib
from datetime import datetime
from typing import Any

import aiohttp
import feedparser
import structlog
from bs4 import BeautifulSoup

from dova.services.source_types import Source, SourceConfig, SourceType, QualityMetrics

logger = structlog.get_logger(__name__)


# Built-in sources (always available)
BUILTIN_SOURCES = {
    "arxiv": Source(
        id="arxiv", user_id="system", name="ArXiv",
        source_type=SourceType.BUILTIN, enabled=True,
        quality=QualityMetrics(query_count=1000, click_count=500),  # high base quality
    ),
    "github": Source(
        id="github", user_id="system", name="GitHub",
        source_type=SourceType.BUILTIN, enabled=True,
        quality=QualityMetrics(query_count=1000, click_count=600),
    ),
    "huggingface": Source(
        id="huggingface", user_id="system", name="HuggingFace",
        source_type=SourceType.BUILTIN, enabled=True,
        quality=QualityMetrics(query_count=1000, click_count=400),
    ),
}


class SourceRegistry:
    """Manages sources per user with quality tracking."""

    def __init__(self, memory_service: Any | None = None):
        self.memory_service = memory_service
        self._cache: dict[str, dict[str, Source]] = {}  # user_id -> {source_id -> Source}
        self._logger = logger.bind(service="source_registry")

    async def get_sources(self, user_id: str, enabled_only: bool = True) -> list[Source]:
        """Get all sources for a user, sorted by quality score."""
        sources = await self._load_user_sources(user_id)

        # Combine with built-ins
        all_sources = list(BUILTIN_SOURCES.values()) + list(sources.values())

        if enabled_only:
            all_sources = [s for s in all_sources if s.enabled]

        # Sort by quality score (highest first)
        return sorted(all_sources, key=lambda s: s.quality.quality_score, reverse=True)

    async def get_source(self, user_id: str, source_id: str) -> Source | None:
        """Get a specific source."""
        if source_id in BUILTIN_SOURCES:
            return BUILTIN_SOURCES[source_id]
        sources = await self._load_user_sources(user_id)
        return sources.get(source_id)

    async def add_source(
        self, user_id: str, name: str, source_type: SourceType, config: SourceConfig
    ) -> Source:
        """Add a custom source for a user."""
        source_id = f"custom_{hashlib.md5(f'{user_id}:{name}'.encode()).hexdigest()[:12]}"

        source = Source(
            id=source_id,
            user_id=user_id,
            name=name,
            source_type=source_type,
            config=config,
        )

        sources = await self._load_user_sources(user_id)
        sources[source_id] = source
        await self._save_user_sources(user_id, sources)

        self._logger.info("source_added", user_id=user_id, source_id=source_id, name=name)
        return source

    async def update_source(self, user_id: str, source_id: str, **updates) -> Source | None:
        """Update a source's configuration."""
        sources = await self._load_user_sources(user_id)
        if source_id not in sources:
            return None

        source = sources[source_id]
        for key, value in updates.items():
            if hasattr(source, key):
                setattr(source, key, value)

        await self._save_user_sources(user_id, sources)
        return source

    async def delete_source(self, user_id: str, source_id: str) -> bool:
        """Delete a custom source."""
        if source_id in BUILTIN_SOURCES:
            return False  # Can't delete built-ins

        sources = await self._load_user_sources(user_id)
        if source_id in sources:
            del sources[source_id]
            await self._save_user_sources(user_id, sources)
            return True
        return False

    async def record_interaction(
        self, user_id: str, source_id: str,
        interaction_type: str,  # "query", "click", "save"
        result_position: int | None = None,
        result_count: int = 0,
    ) -> None:
        """Record an implicit quality signal."""
        source = await self.get_source(user_id, source_id)
        if not source or source.source_type == SourceType.BUILTIN:
            # For built-ins, track per-user quality separately
            sources = await self._load_user_sources(user_id)
            if source_id not in sources and source_id in BUILTIN_SOURCES:
                # Create user-specific override for built-in
                sources[source_id] = Source(
                    id=source_id, user_id=user_id,
                    name=BUILTIN_SOURCES[source_id].name,
                    source_type=SourceType.BUILTIN,
                )
            source = sources.get(source_id)
            if not source:
                return

        # Update metrics
        if interaction_type == "query":
            source.quality.query_count += 1
            source.quality.total_results += result_count
        elif interaction_type == "click":
            source.quality.click_count += 1
            if result_position is not None:
                # Running average of click positions
                n = source.quality.click_count
                source.quality.avg_position_clicked = (
                    (source.quality.avg_position_clicked * (n - 1) + result_position) / n
                )
        elif interaction_type == "save":
            source.quality.save_count += 1

        source.quality.last_used = datetime.utcnow()

        sources = await self._load_user_sources(user_id)
        sources[source.id] = source
        await self._save_user_sources(user_id, sources)

    async def _load_user_sources(self, user_id: str) -> dict[str, Source]:
        """Load sources from cache or memory."""
        if user_id in self._cache:
            return self._cache[user_id]

        if self.memory_service:
            try:
                entries = await self.memory_service.search_memory(
                    f"sources:{user_id}", max_results=1
                )
                if entries and entries[0].content:
                    sources_data = entries[0].content.get("sources", [])
                    sources = {s["id"]: Source.from_dict(s) for s in sources_data}
                    self._cache[user_id] = sources
                    return sources
            except Exception as e:
                self._logger.warning("source_load_error", error=str(e))

        self._cache[user_id] = {}
        return {}

    async def _save_user_sources(self, user_id: str, sources: dict[str, Source]) -> None:
        """Save sources to memory."""
        self._cache[user_id] = sources

        if self.memory_service:
            await self.memory_service.store_long_term(
                f"sources:{user_id}",
                {"sources": [s.to_dict() for s in sources.values()]},
            )


class SourceFetcher:
    """Fetches content from custom sources."""

    def __init__(self):
        self._logger = logger.bind(service="source_fetcher")
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a shared aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def fetch(self, source: Source, query: str) -> list[dict[str, Any]]:
        """Fetch results from a source."""
        if source.source_type == SourceType.BUILTIN:
            return []  # Built-ins handled by existing MCP tools

        if source.source_type == SourceType.WEB_URL:
            return await self._fetch_web(source, query)
        elif source.source_type == SourceType.RSS_FEED:
            return await self._fetch_rss(source, query)
        elif source.source_type == SourceType.API:
            return await self._fetch_api(source, query)

        return []

    async def _fetch_web(self, source: Source, query: str) -> list[dict[str, Any]]:
        """Scrape a web page."""
        if not source.config:
            return []

        try:
            session = self._get_session()
            headers = source.config.headers.copy()
            async with session.get(source.config.url, headers=headers) as resp:
                if resp.status != 200:
                    return []
                html = await resp.text()

            soup = BeautifulSoup(html, "html.parser")

            # Use content selector if provided
            if source.config.content_selector:
                elements = soup.select(source.config.content_selector)
            else:
                elements = soup.find_all(["article", "div", "section"])[:20]

            results = []
            for el in elements:
                title = el.find(["h1", "h2", "h3", "a"])
                if title:
                    results.append({
                        "source": source.id,
                        "title": title.get_text(strip=True)[:200],
                        "url": source.config.url,
                        "description": el.get_text(strip=True)[:500],
                    })

            return results[:10]
        except Exception as e:
            self._logger.warning("web_fetch_error", source=source.id, error=str(e))
            return []

    async def _fetch_rss(self, source: Source, query: str) -> list[dict[str, Any]]:
        """Parse an RSS/Atom feed."""
        if not source.config:
            return []

        try:
            session = self._get_session()
            async with session.get(source.config.url) as resp:
                content = await resp.text()

            feed = feedparser.parse(content)
            query_lower = query.lower()

            results = []
            for entry in feed.entries[:20]:
                title = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))

                # Basic relevance filter
                if query_lower in title.lower() or query_lower in summary.lower():
                    results.append({
                        "source": source.id,
                        "title": title,
                        "url": entry.get("link", ""),
                        "description": summary[:500],
                        "published": entry.get("published", ""),
                    })

            return results[:10]
        except Exception as e:
            self._logger.warning("rss_fetch_error", source=source.id, error=str(e))
            return []

    async def _fetch_api(self, source: Source, query: str) -> list[dict[str, Any]]:
        """Call a custom API endpoint."""
        if not source.config:
            return []

        try:
            headers = source.config.headers.copy()

            # Add authentication
            if source.config.auth_type == "bearer" and source.config.auth_value:
                headers["Authorization"] = f"Bearer {source.config.auth_value}"
            elif source.config.auth_type == "api_key" and source.config.auth_value:
                headers["X-API-Key"] = source.config.auth_value

            # Substitute query in URL
            url = source.config.url.replace("{query}", query)

            session = self._get_session()
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

            # Handle common API response formats
            if isinstance(data, list):
                items = data[:10]
            elif isinstance(data, dict):
                items = data.get("results", data.get("items", data.get("data", [])))[:10]
            else:
                return []

            return [
                {
                    "source": source.id,
                    "title": item.get("title", item.get("name", str(item)[:100])),
                    "url": item.get("url", item.get("link", "")),
                    "description": item.get("description", item.get("summary", ""))[:500],
                }
                for item in items
            ]
        except Exception as e:
            self._logger.warning("api_fetch_error", source=source.id, error=str(e))
            return []
