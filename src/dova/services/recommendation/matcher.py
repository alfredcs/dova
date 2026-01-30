"""
User-content matching for personalized recommendations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class UserProfile:
    """User profile for matching."""

    user_id: str
    interests: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    preferred_sources: list[str] = field(default_factory=list)
    excluded_tags: list[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Match:
    """A content-user match with relevance score."""

    user_id: str
    content_id: str
    score: float
    source: str
    title: str
    url: str
    matched_tags: list[str] = field(default_factory=list)
    reason: str = ""


class UserMatcher:
    """
    Matches content to users based on preferences and embeddings.

    Uses cosine similarity with a configurable threshold.
    """

    def __init__(
        self,
        memory_service: Any | None = None,
        similarity_threshold: float = 0.75,
    ):
        self.memory_service = memory_service
        self.similarity_threshold = similarity_threshold
        self._profile_cache: dict[str, UserProfile] = {}
        self._logger = logger.bind(service="user_matcher")

    async def match_content(
        self,
        content: dict[str, Any],
        user_ids: list[str] | None = None,
    ) -> list[Match]:
        """
        Match a single content item to relevant users.

        Args:
            content: Processed content item with embedding
            user_ids: Specific users to match (None = all with subscriptions)

        Returns:
            List of matches above threshold
        """
        users = await self._get_users(user_ids)
        matches: list[Match] = []

        for user in users:
            score = self._compute_similarity(content, user)

            if score >= self.similarity_threshold:
                matched_tags = self._find_matching_tags(content, user)
                matches.append(
                    Match(
                        user_id=user.user_id,
                        content_id=content["id"],
                        score=score,
                        source=content["source"],
                        title=content["title"],
                        url=content["url"],
                        matched_tags=matched_tags,
                        reason=self._generate_reason(matched_tags, score),
                    )
                )

        self._logger.debug(
            "content_matched",
            content_id=content["id"],
            matches=len(matches),
        )
        return matches

    async def match_batch(
        self,
        contents: list[dict[str, Any]],
        user_ids: list[str] | None = None,
    ) -> dict[str, list[Match]]:
        """
        Match multiple content items to users.

        Returns:
            Dict of user_id -> list of matches
        """
        all_matches: dict[str, list[Match]] = {}

        for content in contents:
            matches = await self.match_content(content, user_ids)
            for match in matches:
                if match.user_id not in all_matches:
                    all_matches[match.user_id] = []
                all_matches[match.user_id].append(match)

        # Sort each user's matches by score
        for user_id in all_matches:
            all_matches[user_id].sort(key=lambda m: m.score, reverse=True)

        return all_matches

    async def _get_users(self, user_ids: list[str] | None) -> list[UserProfile]:
        """Get user profiles for matching."""
        if user_ids:
            return [await self._get_profile(uid) for uid in user_ids]

        # Get all users with active subscriptions
        if self.memory_service:
            # TODO: Query memory service for subscribed users
            pass

        # Return cached profiles
        return list(self._profile_cache.values())

    async def _get_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile."""
        if user_id in self._profile_cache:
            return self._profile_cache[user_id]

        # Try to load from memory service
        profile = UserProfile(user_id=user_id)

        if self.memory_service:
            try:
                data = await self.memory_service.get_user_profile(user_id)
                if data:
                    profile.interests = data.get("interests", [])
                    profile.embedding = data.get("embedding", [])
                    profile.preferred_sources = data.get("preferred_sources", [])
                    profile.excluded_tags = data.get("excluded_tags", [])
            except Exception as e:
                self._logger.warning("profile_load_error", user_id=user_id, error=str(e))

        self._profile_cache[user_id] = profile
        return profile

    def _compute_similarity(self, content: dict[str, Any], user: UserProfile) -> float:
        """Compute relevance score between content and user."""
        score = 0.0

        # Embedding similarity (if both have embeddings)
        content_embedding = content.get("embedding", [])
        if content_embedding and user.embedding:
            embedding_score = self._cosine_similarity(content_embedding, user.embedding)
            score += embedding_score * 0.5

        # Tag/interest overlap
        content_tags = set(t.lower() for t in content.get("tags", []))
        user_interests = set(i.lower() for i in user.interests)

        if content_tags and user_interests:
            overlap = len(content_tags & user_interests)
            max_possible = min(len(content_tags), len(user_interests))
            if max_possible > 0:
                tag_score = overlap / max_possible
                score += tag_score * 0.3

        # Source preference bonus
        if content.get("source") in user.preferred_sources:
            score += 0.2

        # Exclusion penalty
        excluded = set(t.lower() for t in user.excluded_tags)
        if content_tags & excluded:
            score *= 0.5

        return min(score, 1.0)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot / (mag_a * mag_b)

    def _find_matching_tags(self, content: dict[str, Any], user: UserProfile) -> list[str]:
        """Find tags that match user interests."""
        content_tags = set(t.lower() for t in content.get("tags", []))
        user_interests = set(i.lower() for i in user.interests)
        return list(content_tags & user_interests)

    def _generate_reason(self, matched_tags: list[str], score: float) -> str:
        """Generate human-readable match reason."""
        if matched_tags:
            return f"Matches your interests in {', '.join(matched_tags[:3])}"
        elif score > 0.9:
            return "Highly relevant to your research profile"
        else:
            return "May be of interest based on your preferences"

    async def update_profile(
        self,
        user_id: str,
        interests: list[str] | None = None,
        preferred_sources: list[str] | None = None,
        excluded_tags: list[str] | None = None,
    ) -> UserProfile:
        """Update user profile for better matching."""
        profile = await self._get_profile(user_id)

        if interests is not None:
            profile.interests = interests
        if preferred_sources is not None:
            profile.preferred_sources = preferred_sources
        if excluded_tags is not None:
            profile.excluded_tags = excluded_tags

        profile.last_updated = datetime.utcnow()

        # Persist to memory service
        if self.memory_service:
            try:
                await self.memory_service.update_user_profile(
                    user_id,
                    {
                        "interests": profile.interests,
                        "preferred_sources": profile.preferred_sources,
                        "excluded_tags": profile.excluded_tags,
                    },
                )
            except Exception as e:
                self._logger.warning("profile_save_error", user_id=user_id, error=str(e))

        self._profile_cache[user_id] = profile
        return profile
