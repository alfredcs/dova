"""
Unit Tests for DOVA Profiling Agent.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from dova.agents.base import AgentResult, AgentTask
from dova.agents.profiling import ProfilingAgent, UserProfile, TemporalInterests, UserPreferences
from dova.config.providers import LLMRouter


@pytest.fixture
def profiling_agent(
    mock_llm_router: LLMRouter, mock_mcp_client: AsyncMock
) -> ProfilingAgent:
    """Create profiling agent instance for testing."""
    return ProfilingAgent(
        llm_router=mock_llm_router,
        mcp_client=mock_mcp_client,
    )


class TestProfilingAgent:
    """Test cases for ProfilingAgent."""

    def test_profiling_agent_initialization(
        self, mock_llm_router: LLMRouter, mock_mcp_client: AsyncMock
    ) -> None:
        """Test profiling agent initializes correctly."""
        agent = ProfilingAgent(
            llm_router=mock_llm_router,
            mcp_client=mock_mcp_client,
        )

        assert agent.name == "ProfilingAgent"
        assert agent.llm_router == mock_llm_router

    @pytest.mark.asyncio
    async def test_execute_without_user_id(
        self, profiling_agent: ProfilingAgent
    ) -> None:
        """Test execute fails without user_id."""
        task = AgentTask(
            type="get_preferences",
            params={},
            user_id="",  # Empty user_id
        )

        result = await profiling_agent.execute(task)

        assert not result.success
        assert "user" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_get_preferences(self, profiling_agent: ProfilingAgent) -> None:
        """Test getting user preferences."""
        task = AgentTask(
            type="get_preferences",
            params={},
            user_id="test-user-123",
        )

        result = await profiling_agent.execute(task)

        assert result.success
        assert result.data is not None
        assert "preferences" in result.data

    @pytest.mark.asyncio
    async def test_execute_update_preferences(
        self, profiling_agent: ProfilingAgent
    ) -> None:
        """Test updating user preferences."""
        task = AgentTask(
            type="update_preferences",
            params={
                "interests": ["machine learning", "NLP"],
                "expertise_level": "intermediate",
            },
            user_id="test-user-123",
        )

        result = await profiling_agent.execute(task)

        assert result.success

    @pytest.mark.asyncio
    async def test_extract_topics_from_query(
        self, profiling_agent: ProfilingAgent
    ) -> None:
        """Test extracting topics from user query."""
        # Mock LLM response
        profiling_agent.llm_router.complete = AsyncMock(
            return_value=MagicMock(
                content='["transformers", "attention mechanisms", "NLP"]'
            )
        )

        topics = await profiling_agent._extract_topics(
            "I want to understand how multi-head attention works in transformers"
        )

        assert topics is not None
        assert isinstance(topics, list)

    @pytest.mark.asyncio
    async def test_record_interaction(
        self, profiling_agent: ProfilingAgent
    ) -> None:
        """Test recording user interaction."""
        # Mock LLM for topic extraction
        profiling_agent.llm_router.complete = AsyncMock(
            return_value=MagicMock(
                content='["deep learning", "neural networks"]'
            )
        )

        task = AgentTask(
            type="record_interaction",
            params={
                "query": "explain deep learning neural networks",
                "type": "query",
            },
            user_id="test-user-123",
        )

        result = await profiling_agent.execute(task)

        assert result.success
        assert "topics_extracted" in result.data


class TestUserProfile:
    """Test cases for UserProfile model."""

    def test_user_profile_creation(self) -> None:
        """Test creating user profile."""
        profile = UserProfile(
            user_id="test-user",
            preferences=UserPreferences(
                interests=["AI", "ML"],
                expertise_level="advanced",
            ),
        )

        assert profile.user_id == "test-user"
        assert "AI" in profile.preferences.interests
        assert profile.preferences.expertise_level == "advanced"

    def test_user_profile_defaults(self) -> None:
        """Test user profile default values."""
        profile = UserProfile(user_id="test-user")

        assert profile.preferences.interests == []
        assert profile.preferences.expertise_level == "intermediate"
        assert profile.temporal_interests.short_term == []
        assert profile.temporal_interests.medium_term == []
        assert profile.temporal_interests.long_term == []

    def test_user_profile_with_temporal_interests(self) -> None:
        """Test user profile with temporal interests."""
        temporal = TemporalInterests(
            short_term=["LLMs", "GPT"],
            medium_term=["transformers"],
            long_term=["machine learning"],
        )

        profile = UserProfile(
            user_id="test-user",
            temporal_interests=temporal,
        )

        assert len(profile.temporal_interests.short_term) == 2
        assert "LLMs" in profile.temporal_interests.short_term


class TestTemporalInterests:
    """Test cases for TemporalInterests model."""

    def test_temporal_interests_creation(self) -> None:
        """Test creating temporal interests."""
        interests = TemporalInterests(
            short_term=["neural networks"],
            medium_term=["deep learning"],
            long_term=["AI"],
        )

        assert "neural networks" in interests.short_term
        assert "deep learning" in interests.medium_term
        assert "AI" in interests.long_term

    def test_temporal_interests_defaults(self) -> None:
        """Test temporal interests default values."""
        interests = TemporalInterests()

        assert interests.short_term == []
        assert interests.medium_term == []
        assert interests.long_term == []


class TestProfilingAgentMemory:
    """Test memory integration in ProfilingAgent."""

    @pytest.fixture
    def agent(
        self, mock_llm_router: LLMRouter, mock_mcp_client: AsyncMock
    ) -> ProfilingAgent:
        """Create agent for memory tests."""
        return ProfilingAgent(
            llm_router=mock_llm_router,
            mcp_client=mock_mcp_client,
        )

    @pytest.mark.asyncio
    async def test_store_profile(self, agent: ProfilingAgent) -> None:
        """Test storing profile."""
        profile = UserProfile(
            user_id="test-user",
            preferences=UserPreferences(
                interests=["AI"],
                expertise_level="intermediate",
            ),
        )

        # Should not raise
        await agent._save_profile(profile)

    @pytest.mark.asyncio
    async def test_load_profile(self, agent: ProfilingAgent) -> None:
        """Test loading profile."""
        profile = await agent._load_profile("test-user")

        # Should return profile (new or existing)
        assert profile is not None
        assert isinstance(profile, UserProfile)

    def test_profile_to_dict(self, agent: ProfilingAgent) -> None:
        """Test converting profile to dictionary."""
        profile = UserProfile(
            user_id="test-user",
            preferences=UserPreferences(
                interests=["ML"],
                expertise_level="beginner",
            ),
        )

        profile_dict = agent._profile_to_dict(profile)

        assert profile_dict["user_id"] == "test-user"
        assert "preferences" in profile_dict
        assert profile_dict["preferences"]["interests"] == ["ML"]

    def test_dict_to_profile(self, agent: ProfilingAgent) -> None:
        """Test converting dictionary to profile."""
        data = {
            "user_id": "test-user",
            "preferences": {
                "interests": ["DL", "NLP"],
                "expertise_level": "advanced",
            },
            "temporal_interests": {
                "short_term": ["LLMs"],
                "medium_term": [],
                "long_term": [],
            },
            "topic_affinities": {"LLMs": 0.8},
            "interaction_count": 5,
            "created_at": "2024-01-01T00:00:00",
        }

        profile = agent._dict_to_profile(data)

        assert profile.user_id == "test-user"
        assert "DL" in profile.preferences.interests
        assert profile.preferences.expertise_level == "advanced"
