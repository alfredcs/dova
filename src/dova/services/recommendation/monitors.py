"""
Content monitors for ArXiv and HuggingFace.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import aiohttp
import feedparser
import structlog

from dova.jobs.jobs import Job

logger = structlog.get_logger(__name__)


@dataclass
class ContentItem:
    """Normalized content item from any source."""

    id: str
    source: str  # "arxiv", "huggingface", "github"
    title: str
    description: str
    url: str
    authors: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class ArXivMonitor:
    """Monitor ArXiv RSS feeds for new papers."""

    BASE_URL = "http://export.arxiv.org/rss"
    CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE"]

    def __init__(self) -> None:
        self._logger = logger.bind(service="arxiv_monitor")

    async def poll(self, categories: list[str] | None = None) -> list[ContentItem]:
        """
        Poll ArXiv RSS feeds for new papers.

        Args:
            categories: ArXiv categories to poll (defaults to AI-related)

        Returns:
            List of new papers as ContentItems
        """
        categories = categories or self.CATEGORIES
        items: list[ContentItem] = []

        async with aiohttp.ClientSession() as session:
            for category in categories:
                try:
                    papers = await self._fetch_category(session, category)
                    items.extend(papers)
                except Exception as e:
                    self._logger.error("arxiv_fetch_error", category=category, error=str(e))

        self._logger.info("arxiv_poll_complete", count=len(items))
        return items

    async def _fetch_category(
        self,
        session: aiohttp.ClientSession,
        category: str,
    ) -> list[ContentItem]:
        """Fetch papers from a single category RSS feed."""
        url = f"{self.BASE_URL}/{category}"

        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status != 200:
                self._logger.warning(
                    "arxiv_http_error",
                    category=category,
                    status=response.status,
                )
                return []

            content = await response.text()

        feed = feedparser.parse(content)
        items: list[ContentItem] = []

        for entry in feed.entries:
            # Parse ArXiv ID from link
            arxiv_id = entry.link.split("/abs/")[-1] if "/abs/" in entry.link else entry.id

            # Extract authors
            authors = []
            if hasattr(entry, "authors"):
                authors = [a.get("name", "") for a in entry.authors]
            elif hasattr(entry, "author"):
                authors = [entry.author]

            # Extract categories/tags
            tags = [category]
            if hasattr(entry, "tags"):
                tags.extend(t.get("term", "") for t in entry.tags)

            items.append(
                ContentItem(
                    id=f"arxiv:{arxiv_id}",
                    source="arxiv",
                    title=entry.title,
                    description=entry.get("summary", ""),
                    url=entry.link,
                    authors=authors,
                    tags=list(set(tags)),
                    created_at=datetime.utcnow(),
                    metadata={"category": category, "arxiv_id": arxiv_id},
                )
            )

        return items


class HFModelMonitor:
    """Monitor HuggingFace for trending models."""

    API_URL = "https://huggingface.co/api/models"

    def __init__(self) -> None:
        self._logger = logger.bind(service="hf_monitor")

    async def poll(
        self,
        tasks: list[str] | None = None,
        limit: int = 50,
    ) -> list[ContentItem]:
        """
        Poll HuggingFace API for trending models.

        Args:
            tasks: Task types to filter (e.g., "text-generation")
            limit: Max models to fetch

        Returns:
            List of trending models as ContentItems
        """
        tasks = tasks or ["text-generation", "text-classification", "image-classification"]
        items: list[ContentItem] = []

        async with aiohttp.ClientSession() as session:
            for task in tasks:
                try:
                    models = await self._fetch_task_models(session, task, limit // len(tasks))
                    items.extend(models)
                except Exception as e:
                    self._logger.error("hf_fetch_error", task=task, error=str(e))

        self._logger.info("hf_poll_complete", count=len(items))
        return items

    async def _fetch_task_models(
        self,
        session: aiohttp.ClientSession,
        task: str,
        limit: int,
    ) -> list[ContentItem]:
        """Fetch models for a specific task."""
        params = {
            "pipeline_tag": task,
            "sort": "trending",
            "direction": -1,
            "limit": limit,
        }

        async with session.get(
            self.API_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            if response.status != 200:
                self._logger.warning("hf_http_error", task=task, status=response.status)
                return []

            models = await response.json()

        items: list[ContentItem] = []
        for model in models:
            model_id = model.get("id", "")
            items.append(
                ContentItem(
                    id=f"hf:{model_id}",
                    source="huggingface",
                    title=model_id,
                    description=model.get("description", ""),
                    url=f"https://huggingface.co/{model_id}",
                    authors=[model.get("author", "")],
                    tags=model.get("tags", []) + [task],
                    created_at=datetime.utcnow(),
                    metadata={
                        "task": task,
                        "downloads": model.get("downloads", 0),
                        "likes": model.get("likes", 0),
                    },
                )
            )

        return items


# Job handlers for worker registration


async def handle_arxiv_poll(job: Job) -> dict[str, Any]:
    """Handle ArXiv polling job."""
    monitor = ArXivMonitor()
    categories = job.payload.get("categories", ArXivMonitor.CATEGORIES)
    items = await monitor.poll(categories)

    # TODO: Pass items to ContentProcessor and UserMatcher
    # For now, return count
    return {"items_found": len(items), "categories": categories}


async def handle_hf_poll(job: Job) -> dict[str, Any]:
    """Handle HuggingFace polling job."""
    monitor = HFModelMonitor()
    tasks = job.payload.get("tasks", ["text-generation"])
    items = await monitor.poll(tasks)

    # TODO: Pass items to ContentProcessor and UserMatcher
    return {"items_found": len(items), "tasks": tasks}
