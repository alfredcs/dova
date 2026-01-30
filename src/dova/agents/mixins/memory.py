"""Memory mixin for DOVA agents."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dova.services.memory import AgentCoreMemoryService


class MemoryMixin:
    """Adds memory capabilities to agents."""

    memory_service: "AgentCoreMemoryService | None"

    async def remember(
        self,
        content: dict[str, Any],
        user_id: str,
        session_id: str,
        short_term: bool = True,
    ) -> str:
        """Store content to memory."""
        if not self.memory_service:
            return ""
        if short_term:
            return await self.memory_service.store_short_term(
                session_id, content, user_id
            )
        return await self.memory_service.store_long_term(f"user:{user_id}", content)

    async def recall(self, user_id: str, max_results: int = 10) -> list[dict[str, Any]]:
        """Retrieve memories for a user."""
        if not self.memory_service:
            return []
        entries = await self.memory_service.search_memory(
            f"user:{user_id}", max_results
        )
        return [e.content for e in entries]

    async def crystallize(
        self, user_id: str, topic: str, summary: str, source_sessions: list[str]
    ) -> str:
        """Promote short-term memories to permanent knowledge."""
        if not self.memory_service:
            return ""
        return await self.memory_service.promote_to_knowledge(
            user_id, topic, summary, source_sessions
        )
