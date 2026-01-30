"""AgentCore Memory Service for DOVA."""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

import boto3
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MemoryEntry:
    """A memory entry from AgentCore."""

    id: str
    type: str  # "short_term", "long_term", "knowledge"
    content: dict[str, Any]
    created_at: datetime
    expires_at: datetime | None = None
    summary_text: str = ""


class AgentCoreMemoryService:
    """Client for AgentCore Memory API (bedrock-agent-runtime)."""

    def __init__(self, agent_id: str, agent_alias_id: str, region: str = "us-east-1"):
        self.client = boto3.client("bedrock-agent-runtime", region_name=region)
        self.agent_id = agent_id
        self.agent_alias_id = agent_alias_id
        self._logger = logger.bind(service="memory")

    async def store_short_term(
        self, session_id: str, content: dict[str, Any], user_id: str
    ) -> str:
        """Store to session memory via invoke_agent."""
        self._logger.debug("store_short_term", session_id=session_id, user_id=user_id)
        self.client.invoke_agent(
            agentId=self.agent_id,
            agentAliasId=self.agent_alias_id,
            sessionId=session_id,
            memoryId=f"user:{user_id}",
            inputText=f"Remember: {json.dumps(content)}",
        )
        return session_id

    async def store_long_term(self, memory_id: str, content: dict[str, Any]) -> str:
        """Store to persistent memory."""
        self._logger.debug("store_long_term", memory_id=memory_id)
        self.client.invoke_agent(
            agentId=self.agent_id,
            agentAliasId=self.agent_alias_id,
            sessionId=f"knowledge-{uuid4()}",
            memoryId=memory_id,
            inputText=f"Store knowledge: {json.dumps(content)}",
            endSession=True,
        )
        return memory_id

    async def search_memory(
        self, memory_id: str, max_results: int = 10
    ) -> list[MemoryEntry]:
        """Retrieve memory entries for a user."""
        self._logger.debug("search_memory", memory_id=memory_id)
        response = self.client.get_agent_memory(
            agentId=self.agent_id,
            agentAliasId=self.agent_alias_id,
            memoryId=memory_id,
            memoryType="SESSION_SUMMARY",
            maxItems=max_results,
        )
        return [self._parse_entry(e) for e in response.get("memoryContents", [])]

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete memory entry."""
        self.client.delete_agent_memory(
            agentId=self.agent_id,
            agentAliasId=self.agent_alias_id,
            memoryId=memory_id,
        )
        return True

    async def promote_to_knowledge(
        self, user_id: str, topic: str, summary: str, source_sessions: list[str]
    ) -> str:
        """Promote short-term to permanent knowledge."""
        knowledge_id = f"knowledge:{user_id}:{topic.lower().replace(' ', '-')}"
        await self.store_long_term(
            memory_id=knowledge_id,
            content={
                "type": "evolved_knowledge",
                "topic": topic,
                "summary": summary,
                "source_sessions": source_sessions,
                "promoted_at": datetime.utcnow().isoformat(),
            },
        )
        return knowledge_id

    def _parse_entry(self, raw: dict[str, Any]) -> MemoryEntry:
        """Parse raw memory content to MemoryEntry."""
        session = raw.get("sessionSummary", {})
        return MemoryEntry(
            id=session.get("sessionId", ""),
            type="short_term",
            content={"summary": session.get("summaryText", "")},
            created_at=datetime.fromisoformat(
                session.get("sessionStartTime", datetime.utcnow().isoformat())
            ),
            summary_text=session.get("summaryText", ""),
        )
