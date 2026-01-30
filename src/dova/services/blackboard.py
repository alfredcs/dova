"""
Blackboard Service for DOVA.

Shared workspace where agents post insights and build on each other's reasoning.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class PostType(Enum):
    HYPOTHESIS = "hypothesis"  # Initial theory or proposal
    EVIDENCE = "evidence"  # Supporting or refuting evidence
    REFINEMENT = "refinement"  # Improvement to existing post
    QUESTION = "question"  # Request for clarification
    ANSWER = "answer"  # Response to question
    CONSENSUS = "consensus"  # Agreed conclusion


@dataclass
class Vote:
    """A vote on a blackboard post."""

    agent_name: str
    agreement: float  # -1.0 (strongly disagree) to 1.0 (strongly agree)
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BlackboardPost:
    """A contribution to the shared blackboard."""

    id: str
    agent_name: str
    post_type: PostType
    content: str
    confidence: float = 0.5
    references: list[str] = field(default_factory=list)  # IDs of posts this builds on
    votes: list[Vote] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def agreement_score(self) -> float:
        """Average agreement from votes."""
        if not self.votes:
            return 0.0
        return sum(v.agreement for v in self.votes) / len(self.votes)

    @property
    def weighted_confidence(self) -> float:
        """Confidence adjusted by agreement."""
        return self.confidence * (1 + self.agreement_score) / 2


class Blackboard:
    """Shared workspace for collaborative agent reasoning."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid4())
        self._posts: dict[str, BlackboardPost] = {}
        self._logger = logger.bind(session_id=self.session_id)

    async def post(
        self,
        agent_name: str,
        post_type: PostType,
        content: str,
        confidence: float = 0.5,
        references: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Post an insight to the blackboard.

        Args:
            agent_name: Name of the posting agent
            post_type: Type of post
            content: The insight content
            confidence: Agent's confidence in this insight
            references: IDs of posts this builds upon

        Returns:
            Post ID
        """
        post_id = f"post_{uuid4().hex[:8]}"
        post = BlackboardPost(
            id=post_id,
            agent_name=agent_name,
            post_type=post_type,
            content=content,
            confidence=confidence,
            references=references or [],
            metadata=metadata or {},
        )
        self._posts[post_id] = post

        self._logger.info(
            "blackboard_post",
            post_id=post_id,
            agent=agent_name,
            post_type=post_type.value,
        )
        return post_id

    async def vote(
        self,
        post_id: str,
        agent_name: str,
        agreement: float,
        reasoning: str = "",
    ) -> None:
        """
        Vote on a post.

        Args:
            post_id: ID of post to vote on
            agent_name: Voting agent's name
            agreement: -1.0 to 1.0 agreement score
            reasoning: Optional explanation
        """
        if post_id not in self._posts:
            raise ValueError(f"Post {post_id} not found")

        self._posts[post_id].votes.append(
            Vote(
                agent_name=agent_name,
                agreement=max(-1.0, min(1.0, agreement)),
                reasoning=reasoning,
            )
        )

    async def get_context(
        self,
        agent_name: str | None = None,
        post_types: list[PostType] | None = None,
        min_confidence: float = 0.0,
        exclude_own: bool = False,
        max_posts: int = 20,
    ) -> list[BlackboardPost]:
        """
        Get relevant posts for an agent's context.

        Args:
            agent_name: Filter by agent (for exclude_own)
            post_types: Filter by post types
            min_confidence: Minimum weighted confidence
            exclude_own: Exclude posts from the requesting agent
            max_posts: Maximum posts to return

        Returns:
            List of relevant posts, sorted by weighted confidence
        """
        posts = list(self._posts.values())

        if post_types:
            posts = [p for p in posts if p.post_type in post_types]

        if exclude_own and agent_name:
            posts = [p for p in posts if p.agent_name != agent_name]

        posts = [p for p in posts if p.weighted_confidence >= min_confidence]

        # Sort by weighted confidence descending
        posts.sort(key=lambda p: p.weighted_confidence, reverse=True)

        return posts[:max_posts]

    async def get_thread(self, post_id: str) -> list[BlackboardPost]:
        """Get a post and all posts that reference it."""
        if post_id not in self._posts:
            return []

        thread = [self._posts[post_id]]
        for post in self._posts.values():
            if post_id in post.references:
                thread.append(post)

        return sorted(thread, key=lambda p: p.timestamp)

    async def synthesize(self, llm_func: Any = None) -> dict[str, Any]:
        """
        Synthesize all posts into coherent conclusions.

        Args:
            llm_func: Optional LLM function for synthesis

        Returns:
            Synthesis with key conclusions and confidence
        """
        hypotheses = [
            p for p in self._posts.values() if p.post_type == PostType.HYPOTHESIS
        ]
        evidence = [
            p for p in self._posts.values() if p.post_type == PostType.EVIDENCE
        ]
        consensuses = [
            p for p in self._posts.values() if p.post_type == PostType.CONSENSUS
        ]

        # Simple synthesis without LLM
        synthesis = {
            "total_posts": len(self._posts),
            "hypotheses": [
                {
                    "content": h.content,
                    "confidence": h.weighted_confidence,
                    "agreement": h.agreement_score,
                }
                for h in sorted(
                    hypotheses, key=lambda x: x.weighted_confidence, reverse=True
                )
            ],
            "key_evidence": [
                {"content": e.content, "supports": e.references}
                for e in sorted(
                    evidence, key=lambda x: x.weighted_confidence, reverse=True
                )[:5]
            ],
            "consensuses": [c.content for c in consensuses],
            "overall_confidence": (
                sum(p.weighted_confidence for p in self._posts.values())
                / len(self._posts)
                if self._posts
                else 0.0
            ),
        }

        return synthesis

    def clear(self) -> None:
        """Clear all posts from the blackboard."""
        self._posts.clear()
