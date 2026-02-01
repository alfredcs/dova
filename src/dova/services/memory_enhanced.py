"""
Enhanced memory service with semantic search and diversity ranking.

Provides multi-tier memory storage with embedding-based retrieval
and MMR (Maximal Marginal Relevance) for diverse results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

from dova.utils.cache import Cache

logger = structlog.get_logger(__name__)


class MemoryType(Enum):
    """Types of memory storage."""

    SHORT_TERM = "short_term"  # Session-scoped, temporary
    LONG_TERM = "long_term"  # Persistent across sessions
    PROCEDURAL = "procedural"  # How-to knowledge, skills


@dataclass
class EnhancedMemoryEntry:
    """A memory entry with optional embedding."""

    id: str
    type: MemoryType
    content: dict[str, Any]
    embedding: list[float] | None = None
    importance: float = 0.5  # 0.0 to 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime | None = None
    access_count: int = 0
    user_id: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "embedding": self.embedding,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat() if self.accessed_at else None,
            "access_count": self.access_count,
            "user_id": self.user_id,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnhancedMemoryEntry":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            type=MemoryType(data["type"]),
            content=data["content"],
            embedding=data.get("embedding"),
            importance=data.get("importance", 0.5),
            created_at=datetime.fromisoformat(data["created_at"]),
            accessed_at=(
                datetime.fromisoformat(data["accessed_at"])
                if data.get("accessed_at")
                else None
            ),
            access_count=data.get("access_count", 0),
            user_id=data.get("user_id"),
            tags=data.get("tags", []),
        )


@dataclass
class SearchResult:
    """Result from semantic memory search."""

    entry: EnhancedMemoryEntry
    score: float  # Similarity score
    relevance_rank: int = 0


class EnhancedMemoryService:
    """
    Enhanced memory service with semantic search capabilities.

    Supports multiple memory types, embedding-based retrieval,
    and MMR reranking for result diversity.
    """

    def __init__(
        self,
        cache: Cache,
        llm_router: Any | None = None,  # LLMRouter for embeddings
        mmr_lambda: float = 0.5,
        embedding_cache_ttl: int = 3600,
    ):
        self.cache = cache
        self.llm_router = llm_router
        self.mmr_lambda = mmr_lambda
        self.embedding_cache_ttl = embedding_cache_ttl
        self._logger = logger.bind(service="memory_enhanced")

    async def store(
        self,
        memory_type: MemoryType,
        content: dict[str, Any],
        importance: float = 0.5,
        user_id: str | None = None,
        tags: list[str] | None = None,
        generate_embedding: bool = True,
    ) -> str:
        """
        Store a memory entry.

        Args:
            memory_type: Type of memory to store
            content: Content to store
            importance: Importance score (0.0 to 1.0)
            user_id: Optional user identifier
            tags: Optional tags for categorization
            generate_embedding: Whether to generate embedding

        Returns:
            Memory entry ID
        """
        entry_id = str(uuid4())

        # Generate embedding if requested and LLM router available
        embedding = None
        if generate_embedding and self.llm_router:
            embedding = await self._generate_embedding(content)

        entry = EnhancedMemoryEntry(
            id=entry_id,
            type=memory_type,
            content=content,
            embedding=embedding,
            importance=importance,
            user_id=user_id,
            tags=tags or [],
        )

        await self._store_entry(entry)

        self._logger.info(
            "memory_stored",
            entry_id=entry_id,
            memory_type=memory_type.value,
            has_embedding=embedding is not None,
        )

        return entry_id

    async def get(self, entry_id: str) -> EnhancedMemoryEntry | None:
        """
        Retrieve a memory entry by ID.

        Args:
            entry_id: Entry identifier

        Returns:
            Entry if found, None otherwise
        """
        cache_key = self._entry_key(entry_id)
        data = await self.cache.get(cache_key)

        if data is None:
            return None

        entry = EnhancedMemoryEntry.from_dict(data)

        # Update access metadata
        entry.accessed_at = datetime.utcnow()
        entry.access_count += 1
        await self._store_entry(entry)

        return entry

    async def search_semantic(
        self,
        query: str,
        user_id: str | None = None,
        top_k: int = 10,
        use_mmr: bool = True,
        memory_types: list[MemoryType] | None = None,
    ) -> list[SearchResult]:
        """
        Search memory using semantic similarity.

        Args:
            query: Search query
            user_id: Optional user to filter by
            top_k: Number of results to return
            use_mmr: Use MMR for diversity
            memory_types: Filter by memory types

        Returns:
            List of search results
        """
        # Generate query embedding
        if not self.llm_router:
            self._logger.warning("semantic_search_unavailable", reason="no llm_router")
            return []

        query_embedding = await self._generate_embedding({"text": query})
        if query_embedding is None:
            return []

        # Get candidate entries
        candidates = await self._get_candidates(user_id, memory_types)

        if not candidates:
            return []

        # Calculate similarity scores
        scored_results: list[SearchResult] = []
        for entry in candidates:
            if entry.embedding is None:
                continue

            score = self._cosine_similarity(query_embedding, entry.embedding)
            scored_results.append(SearchResult(entry=entry, score=score))

        # Sort by score
        scored_results.sort(key=lambda x: x.score, reverse=True)

        # Apply MMR if requested
        if use_mmr and len(scored_results) > top_k:
            scored_results = self._mmr_rerank(
                scored_results,
                query_embedding,
                top_k,
            )

        # Take top_k results
        results = scored_results[:top_k]

        # Assign ranks
        for i, result in enumerate(results):
            result.relevance_rank = i + 1

        self._logger.debug(
            "semantic_search_complete",
            query_len=len(query),
            candidates=len(candidates),
            results=len(results),
        )

        return results

    async def search_by_tags(
        self,
        tags: list[str],
        user_id: str | None = None,
        top_k: int = 10,
    ) -> list[EnhancedMemoryEntry]:
        """
        Search memory by tags.

        Args:
            tags: Tags to search for
            user_id: Optional user filter
            top_k: Maximum results

        Returns:
            List of matching entries
        """
        candidates = await self._get_candidates(user_id, None)
        tag_set = set(tag.lower() for tag in tags)

        matches = [
            entry for entry in candidates
            if tag_set.intersection(tag.lower() for tag in entry.tags)
        ]

        # Sort by importance and recency
        matches.sort(
            key=lambda e: (e.importance, e.created_at.timestamp()),
            reverse=True,
        )

        return matches[:top_k]

    async def delete(self, entry_id: str) -> bool:
        """
        Delete a memory entry.

        Args:
            entry_id: Entry identifier

        Returns:
            True if deleted, False if not found
        """
        cache_key = self._entry_key(entry_id)
        deleted = await self.cache.delete(cache_key)

        if deleted:
            self._logger.info("memory_deleted", entry_id=entry_id)

        return deleted

    async def promote_importance(
        self,
        entry_id: str,
        boost: float = 0.1,
    ) -> EnhancedMemoryEntry | None:
        """
        Increase importance of a memory entry.

        Args:
            entry_id: Entry identifier
            boost: Amount to increase importance

        Returns:
            Updated entry or None if not found
        """
        entry = await self.get(entry_id)
        if entry is None:
            return None

        entry.importance = min(1.0, entry.importance + boost)
        await self._store_entry(entry)

        return entry

    def _mmr_rerank(
        self,
        results: list[SearchResult],
        query_embedding: list[float],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Rerank results using Maximal Marginal Relevance.

        Balances relevance with diversity to avoid redundant results.

        Args:
            results: Initial scored results
            query_embedding: Query embedding for relevance
            top_k: Number of results to select

        Returns:
            MMR-reranked results
        """
        if len(results) <= 1:
            return results

        selected: list[SearchResult] = []
        remaining = list(results)

        while len(selected) < top_k and remaining:
            best_idx = -1
            best_score = float("-inf")

            for i, candidate in enumerate(remaining):
                if candidate.entry.embedding is None:
                    continue

                # Relevance to query
                relevance = candidate.score

                # Maximum similarity to already selected
                max_sim_selected = 0.0
                if selected:
                    for sel in selected:
                        if sel.entry.embedding is not None:
                            sim = self._cosine_similarity(
                                candidate.entry.embedding,
                                sel.entry.embedding,
                            )
                            max_sim_selected = max(max_sim_selected, sim)

                # MMR score: balance relevance and diversity
                mmr_score = (
                    self.mmr_lambda * relevance
                    - (1 - self.mmr_lambda) * max_sim_selected
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
            else:
                break

        return selected

    async def _generate_embedding(
        self,
        content: dict[str, Any],
    ) -> list[float] | None:
        """Generate embedding for content."""
        if not self.llm_router:
            return None

        try:
            # Convert content to text for embedding
            text = self._content_to_text(content)

            # Check cache first
            cache_key = f"embedding:{hash(text)}"
            cached = await self.cache.get(cache_key)
            if cached:
                return cached

            # Generate embedding
            embeddings = await self.llm_router.embed([text])
            if embeddings:
                embedding = embeddings[0]
                # Cache the embedding
                await self.cache.set(
                    cache_key,
                    embedding,
                    ttl=self.embedding_cache_ttl,
                )
                return embedding

        except Exception as e:
            self._logger.warning("embedding_generation_failed", error=str(e))

        return None

    def _content_to_text(self, content: dict[str, Any]) -> str:
        """Convert content dict to text for embedding."""
        if "text" in content:
            return str(content["text"])
        if "summary" in content:
            return str(content["summary"])
        # Fallback: concatenate all string values
        parts = [str(v) for v in content.values() if isinstance(v, str)]
        return " ".join(parts)

    def _cosine_similarity(
        self,
        vec1: list[float],
        vec2: list[float],
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def _get_candidates(
        self,
        user_id: str | None,
        memory_types: list[MemoryType] | None,
    ) -> list[EnhancedMemoryEntry]:
        """Get candidate entries for search."""
        # This is a simplified implementation
        # In production, you'd use a proper index or vector store
        index_key = self._index_key(user_id)
        entry_ids = await self.cache.get(index_key) or []

        candidates = []
        for entry_id in entry_ids:
            entry = await self.get(entry_id)
            if entry is None:
                continue
            if memory_types and entry.type not in memory_types:
                continue
            candidates.append(entry)

        return candidates

    def _entry_key(self, entry_id: str) -> str:
        """Generate cache key for an entry."""
        return f"memory:{entry_id}"

    def _index_key(self, user_id: str | None) -> str:
        """Generate cache key for entry index."""
        if user_id:
            return f"memory_index:{user_id}"
        return "memory_index:global"

    async def _store_entry(self, entry: EnhancedMemoryEntry) -> None:
        """Store entry and update index."""
        # Store entry
        cache_key = self._entry_key(entry.id)
        ttl = None if entry.type == MemoryType.LONG_TERM else 86400  # 24h for short-term
        await self.cache.set(cache_key, entry.to_dict(), ttl=ttl)

        # Update index
        index_key = self._index_key(entry.user_id)
        entry_ids = await self.cache.get(index_key) or []
        if entry.id not in entry_ids:
            entry_ids.append(entry.id)
            await self.cache.set(index_key, entry_ids, ttl=None)
