"""
Content processor for normalizing and embedding content.
"""

import hashlib
from typing import Any

import structlog

from dova.config.providers import LLMRouter, TaskType
from dova.services.recommendation.monitors import ContentItem

logger = structlog.get_logger(__name__)


class ContentProcessor:
    """
    Processes content items for recommendation matching.

    Normalizes content and generates embeddings for similarity matching.
    """

    def __init__(
        self,
        llm_router: LLMRouter | None = None,
        cache_embeddings: bool = True,
    ):
        self.llm_router = llm_router
        self.cache_embeddings = cache_embeddings
        self._embedding_cache: dict[str, list[float]] = {}
        self._logger = logger.bind(service="content_processor")

    async def process(self, item: ContentItem) -> dict[str, Any]:
        """
        Process a content item for matching.

        Returns:
            Processed item with embedding and normalized text
        """
        # Generate content hash for deduplication
        content_hash = self._generate_hash(item)

        # Normalize text
        normalized_text = self._normalize_text(item)

        # Generate embedding
        embedding = await self._get_embedding(item.id, normalized_text)

        return {
            "id": item.id,
            "source": item.source,
            "title": item.title,
            "url": item.url,
            "content_hash": content_hash,
            "normalized_text": normalized_text,
            "embedding": embedding,
            "tags": item.tags,
            "metadata": item.metadata,
        }

    async def process_batch(self, items: list[ContentItem]) -> list[dict[str, Any]]:
        """Process multiple items efficiently."""
        processed = []
        for item in items:
            try:
                result = await self.process(item)
                processed.append(result)
            except Exception as e:
                self._logger.error("process_error", item_id=item.id, error=str(e))
        return processed

    def _generate_hash(self, item: ContentItem) -> str:
        """Generate content hash for deduplication."""
        content = f"{item.title}:{item.description[:500]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _normalize_text(self, item: ContentItem) -> str:
        """Normalize content text for embedding."""
        parts = [item.title]

        if item.description:
            # Truncate description for embedding
            desc = item.description[:1000]
            parts.append(desc)

        if item.tags:
            parts.append(f"Tags: {', '.join(item.tags[:10])}")

        if item.authors:
            parts.append(f"Authors: {', '.join(item.authors[:5])}")

        return " | ".join(parts)

    async def _get_embedding(self, item_id: str, text: str) -> list[float]:
        """Get embedding for text, with caching."""
        if self.cache_embeddings and item_id in self._embedding_cache:
            return self._embedding_cache[item_id]

        if self.llm_router:
            try:
                # Use LLM router for embedding if available
                embedding = await self._generate_embedding(text)
            except Exception as e:
                self._logger.warning("embedding_error", error=str(e))
                embedding = self._simple_embedding(text)
        else:
            # Fallback to simple term-frequency embedding
            embedding = self._simple_embedding(text)

        if self.cache_embeddings:
            self._embedding_cache[item_id] = embedding

        return embedding

    async def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding using LLM router."""
        # This would use the embedding model from the LLM router
        # For now, use simple embedding as placeholder
        return self._simple_embedding(text)

    def _simple_embedding(self, text: str, dim: int = 64) -> list[float]:
        """
        Simple term-frequency based embedding.

        A lightweight fallback when no embedding model is available.
        """
        # Normalize text
        words = text.lower().split()

        # Hash words to fixed dimensions
        embedding = [0.0] * dim
        for word in words:
            idx = hash(word) % dim
            embedding[idx] += 1.0

        # Normalize
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self._logger.debug("cache_cleared")
