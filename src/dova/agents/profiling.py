"""
User Profiling Agent for DOVA.

Manages user profiles with:
- Explicit preferences (declared interests)
- Implicit preferences (inferred from behavior)
- Temporal preferences (short/medium/long-term interests)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dova.agents.user_model import UserModel

import structlog

from dova.agents.base import AgentResult, AgentTask, BaseAgent
from dova.config.providers import LLMRouter, TaskType
from dova.utils.cache import Cache
from dova.utils.metrics import MetricsCollector

logger = structlog.get_logger(__name__)


@dataclass
class UserPreferences:
    """User's explicit preferences."""

    interests: list[str] = field(default_factory=list)
    preferred_sources: list[str] = field(default_factory=list)
    expertise_level: str = "intermediate"
    output_format: str = "detailed"
    notification_frequency: str = "daily"


@dataclass
class TemporalInterests:
    """User interests across time horizons."""

    short_term: list[str] = field(default_factory=list)  # Last 7 days
    medium_term: list[str] = field(default_factory=list)  # Last 30 days
    long_term: list[str] = field(default_factory=list)  # Historical baseline


@dataclass
class UserProfile:
    """Complete user profile."""

    user_id: str
    preferences: UserPreferences = field(default_factory=UserPreferences)
    temporal_interests: TemporalInterests = field(default_factory=TemporalInterests)
    topic_affinities: dict[str, float] = field(default_factory=dict)
    interaction_count: int = 0
    last_active: datetime | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class ProfilingAgent(BaseAgent):
    """
    User Profiling Agent for personalization.

    Uses AgentCore Memory for persistent profile storage
    with multiple memory strategies.
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        cache: Cache | None = None,
        mcp_client: Any | None = None,
        metrics: MetricsCollector | None = None,
        memory_service: Any | None = None,
    ):
        super().__init__(llm_router, mcp_client, metrics, memory_service=memory_service)
        self.cache = cache
        self._profiles: dict[str, UserProfile] = {}

    async def to_user_model(self, user_id: str) -> "UserModel":
        """
        Convert UserProfile to UserModel for ThinkingOrchestrator.

        Maps existing profile data to the richer UserModel format
        used by the deliberation-first orchestrator.
        """
        from dova.agents.user_model import ExpertiseLevel, ResponseDepth, UserModel

        profile = await self._load_profile(user_id)

        # Infer expertise from topic affinities
        expertise_areas: dict[str, ExpertiseLevel] = {}
        for topic, affinity in profile.topic_affinities.items():
            if affinity >= 0.8:
                expertise_areas[topic] = ExpertiseLevel.EXPERT
            elif affinity >= 0.5:
                expertise_areas[topic] = ExpertiseLevel.INTERMEDIATE
            elif affinity > 0:
                expertise_areas[topic] = ExpertiseLevel.BEGINNER

        # Map output format to response depth
        format_to_depth = {
            "detailed": ResponseDepth.DETAILED,
            "brief": ResponseDepth.BRIEF,
            "standard": ResponseDepth.STANDARD,
        }
        preferred_depth = format_to_depth.get(
            profile.preferences.output_format, ResponseDepth.STANDARD
        )

        # Infer formality from expertise level
        formality = "technical" if profile.preferences.expertise_level == "expert" else "standard"

        return UserModel(
            user_id=user_id,
            expertise_areas=expertise_areas,
            preferred_depth=preferred_depth,
            prefers_code_examples=True,
            prefers_citations=True,
            formality=formality,
            current_goals=profile.temporal_interests.short_term[:3],
            entities_of_interest={},
            created_at=profile.created_at,
        )

    @property
    def system_prompt(self) -> str:
        return """You are a User Profiling Agent that analyzes user behavior to build and maintain personalized profiles.

Your responsibilities:
1. Extract interests from user queries and interactions
2. Track topic affinities over time
3. Distinguish between short-term curiosity and long-term interests
4. Provide personalization recommendations

When analyzing interactions:
- Identify explicit interest signals (stated preferences)
- Detect implicit signals (topics queried, content saved)
- Note expertise level from question complexity
- Track temporal patterns (time of day, frequency)"""

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute profiling task."""
        start_time = time.time()

        try:
            task_type = task.type
            user_id = task.params.get("user_id") or task.user_id

            if not user_id:
                return self._wrap_result(task, False, error="No user_id provided")

            self._logger.info("profiling_task", task_type=task_type, user_id=user_id)

            if task_type == "get_preferences":
                result = await self._get_preferences(user_id)
            elif task_type == "update_preferences":
                result = await self._update_preferences(user_id, task.params)
            elif task_type == "record_interaction":
                result = await self._record_interaction(user_id, task.params)
            elif task_type == "get_recommendations":
                result = await self._get_recommendations(user_id)
            else:
                return self._wrap_result(task, False, error=f"Unknown task type: {task_type}")

            execution_time = (time.time() - start_time) * 1000
            return self._wrap_result(
                task,
                True,
                data=result,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            self._logger.exception("profiling_error", error=str(e))
            return self._wrap_result(
                task,
                False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _get_preferences(self, user_id: str) -> dict[str, Any]:
        """Get user preferences."""
        profile = await self._load_profile(user_id)
        return {
            "preferences": {
                "interests": profile.preferences.interests,
                "preferred_sources": profile.preferences.preferred_sources,
                "expertise_level": profile.preferences.expertise_level,
                "output_format": profile.preferences.output_format,
            },
            "temporal_interests": {
                "short_term": profile.temporal_interests.short_term,
                "medium_term": profile.temporal_interests.medium_term,
                "long_term": profile.temporal_interests.long_term,
            },
            "topic_affinities": profile.topic_affinities,
        }

    async def _update_preferences(
        self,
        user_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Update user preferences."""
        profile = await self._load_profile(user_id)

        # Update explicit preferences
        if "interests" in params:
            profile.preferences.interests = params["interests"]
        if "preferred_sources" in params:
            profile.preferences.preferred_sources = params["preferred_sources"]
        if "expertise_level" in params:
            profile.preferences.expertise_level = params["expertise_level"]
        if "output_format" in params:
            profile.preferences.output_format = params["output_format"]

        await self._save_profile(profile)

        return {"status": "updated", "user_id": user_id}

    async def _record_interaction(
        self,
        user_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Record user interaction and update implicit preferences."""
        profile = await self._load_profile(user_id)

        query = params.get("query", "")
        interaction_type = params.get("type", "query")
        topics = params.get("topics", [])

        # Extract topics from query if not provided
        if query and not topics:
            topics = await self._extract_topics(query)

        # Update topic affinities
        for topic in topics:
            current = profile.topic_affinities.get(topic, 0.0)
            # Decay existing + add new signal
            profile.topic_affinities[topic] = min(current * 0.95 + 0.1, 1.0)

        # Update temporal interests
        profile.temporal_interests.short_term = topics[:5]
        for topic in topics:
            if topic not in profile.temporal_interests.medium_term:
                profile.temporal_interests.medium_term.append(topic)
                profile.temporal_interests.medium_term = profile.temporal_interests.medium_term[-10:]

        # Check for long-term interest promotion
        high_affinity_topics = [
            t for t, s in profile.topic_affinities.items() if s > 0.7
        ]
        for topic in high_affinity_topics:
            if topic not in profile.temporal_interests.long_term:
                profile.temporal_interests.long_term.append(topic)
                profile.temporal_interests.long_term = profile.temporal_interests.long_term[-20:]

        profile.interaction_count += 1
        profile.last_active = datetime.utcnow()

        await self._save_profile(profile)

        return {
            "status": "recorded",
            "topics_extracted": topics,
            "interaction_count": profile.interaction_count,
        }

    async def _get_recommendations(self, user_id: str) -> dict[str, Any]:
        """Get personalized recommendations based on profile."""
        profile = await self._load_profile(user_id)

        # Build recommendation context
        recommendation_prompt = f"""Based on this user profile, suggest relevant research directions:

Interests: {profile.preferences.interests}
Recent topics (short-term): {profile.temporal_interests.short_term}
Ongoing interests (medium-term): {profile.temporal_interests.medium_term}
Core interests (long-term): {profile.temporal_interests.long_term}
Topic affinities: {dict(sorted(profile.topic_affinities.items(), key=lambda x: x[1], reverse=True)[:10])}
Expertise level: {profile.preferences.expertise_level}

Provide recommendations for:
1. Papers to read
2. Repositories to explore
3. Models to try
4. Emerging topics in their areas

Format as JSON with keys: papers, repos, models, emerging_topics"""

        recommendations = await self.think(
            recommendation_prompt,
            task_type=TaskType.REASONING,
            temperature=0.7,
        )

        import json

        try:
            if "```json" in recommendations:
                recommendations = recommendations.split("```json")[1].split("```")[0]
            elif "```" in recommendations:
                recommendations = recommendations.split("```")[1].split("```")[0]
            return json.loads(recommendations.strip())
        except json.JSONDecodeError:
            return {"raw_recommendations": recommendations}

    async def _extract_topics(self, query: str) -> list[str]:
        """Extract topics from a query using LLM."""
        extraction_prompt = f"""Extract key topics from this research query:

Query: "{query}"

Return a JSON array of 1-5 topic strings.
Example: ["machine learning", "transformers", "NLP"]"""

        response = await self.think(
            extraction_prompt,
            task_type=TaskType.CLASSIFICATION,
            temperature=0.3,
            max_tokens=100,
        )

        import json

        try:
            if "[" in response:
                start = response.index("[")
                end = response.rindex("]") + 1
                return json.loads(response[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: simple extraction
        return [query.strip()[:50]]

    async def _load_profile(self, user_id: str) -> UserProfile:
        """Load user profile from cache/memory."""
        # Check in-memory cache
        if user_id in self._profiles:
            return self._profiles[user_id]

        # Check external cache
        if self.cache:
            cached = await self.cache.get(f"profile:{user_id}")
            if cached:
                profile = self._dict_to_profile(cached)
                self._profiles[user_id] = profile
                return profile

        # Check AgentCore Memory
        if self.memory_service:
            try:
                memory_data = await self._load_from_memory(user_id)
                if memory_data:
                    profile = self._dict_to_profile(memory_data)
                    self._profiles[user_id] = profile
                    return profile
            except Exception as e:
                self._logger.warning("memory_load_error", user_id=user_id, error=str(e))

        # Create new profile
        profile = UserProfile(user_id=user_id)
        self._profiles[user_id] = profile
        return profile

    async def _save_profile(self, profile: UserProfile) -> None:
        """Save user profile to cache/memory."""
        self._profiles[profile.user_id] = profile
        profile_dict = self._profile_to_dict(profile)

        # Save to external cache
        if self.cache:
            await self.cache.set(f"profile:{profile.user_id}", profile_dict, ttl=86400)

        # Save to AgentCore Memory
        if self.memory_service:
            try:
                await self._save_to_memory(profile.user_id, profile_dict)
            except Exception as e:
                self._logger.warning("memory_save_error", user_id=profile.user_id, error=str(e))

    async def _load_from_memory(self, user_id: str) -> dict[str, Any] | None:
        """Load profile from AgentCore Memory."""
        if not self.memory_service:
            return None
        entries = await self.memory_service.search_memory(
            f"profile:{user_id}", max_results=1
        )
        return entries[0].content if entries else None

    async def _save_to_memory(self, user_id: str, profile_dict: dict[str, Any]) -> None:
        """Save profile to AgentCore Memory."""
        if self.memory_service:
            await self.memory_service.store_long_term(
                f"profile:{user_id}", profile_dict
            )

    def _profile_to_dict(self, profile: UserProfile) -> dict[str, Any]:
        """Convert UserProfile to dictionary."""
        return {
            "user_id": profile.user_id,
            "preferences": {
                "interests": profile.preferences.interests,
                "preferred_sources": profile.preferences.preferred_sources,
                "expertise_level": profile.preferences.expertise_level,
                "output_format": profile.preferences.output_format,
                "notification_frequency": profile.preferences.notification_frequency,
            },
            "temporal_interests": {
                "short_term": profile.temporal_interests.short_term,
                "medium_term": profile.temporal_interests.medium_term,
                "long_term": profile.temporal_interests.long_term,
            },
            "topic_affinities": profile.topic_affinities,
            "interaction_count": profile.interaction_count,
            "last_active": profile.last_active.isoformat() if profile.last_active else None,
            "created_at": profile.created_at.isoformat(),
        }

    def _dict_to_profile(self, data: dict[str, Any]) -> UserProfile:
        """Convert dictionary to UserProfile."""
        prefs = data.get("preferences", {})
        temporal = data.get("temporal_interests", {})

        return UserProfile(
            user_id=data["user_id"],
            preferences=UserPreferences(
                interests=prefs.get("interests", []),
                preferred_sources=prefs.get("preferred_sources", []),
                expertise_level=prefs.get("expertise_level", "intermediate"),
                output_format=prefs.get("output_format", "detailed"),
                notification_frequency=prefs.get("notification_frequency", "daily"),
            ),
            temporal_interests=TemporalInterests(
                short_term=temporal.get("short_term", []),
                medium_term=temporal.get("medium_term", []),
                long_term=temporal.get("long_term", []),
            ),
            topic_affinities=data.get("topic_affinities", {}),
            interaction_count=data.get("interaction_count", 0),
            last_active=datetime.fromisoformat(data["last_active"]) if data.get("last_active") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
        )
