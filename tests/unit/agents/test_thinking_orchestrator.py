"""
Unit tests for ThinkingOrchestrator.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from dova.agents.thinking_orchestrator import (
    ThinkingOrchestrator,
    ActionDecision,
    Deliberation,
    ToolConsideration,
    DELIBERATION_PROMPT,
)
from dova.agents.user_model import UserModel, ExpertiseLevel, ResponseDepth
from dova.agents.conversation_context import ConversationContext
from dova.agents.base import AgentTask, AgentResult


@pytest.fixture
def mock_llm_router():
    """Create a mock LLM router."""
    router = MagicMock()
    router.complete = AsyncMock()
    return router


@pytest.fixture
def orchestrator(mock_llm_router):
    """Create a ThinkingOrchestrator instance."""
    return ThinkingOrchestrator(
        llm_router=mock_llm_router,
        agents={},
        mcp_client=None,
    )


class TestThinkingOrchestrator:
    """Tests for ThinkingOrchestrator."""

    def test_initialization(self, orchestrator):
        """Test orchestrator initializes correctly."""
        assert orchestrator.name == "ThinkingOrchestrator"
        assert orchestrator.agents == {}
        assert orchestrator._user_models == {}
        assert orchestrator._contexts == {}

    def test_register_agent(self, orchestrator):
        """Test agent registration."""
        mock_agent = MagicMock()
        orchestrator.register_agent("research", mock_agent)
        assert "research" in orchestrator.agents
        assert orchestrator.agents["research"] == mock_agent

    @pytest.mark.asyncio
    async def test_execute_without_query(self, orchestrator):
        """Test execute returns error without query."""
        task = AgentTask(type="query", params={})
        result = await orchestrator.execute(task)

        assert not result.success
        assert "No query provided" in result.error

    @pytest.mark.asyncio
    async def test_load_user_model_creates_new(self, orchestrator):
        """Test loading user model creates new if not exists."""
        user_model = await orchestrator._load_user_model("test-user")

        assert user_model.user_id == "test-user"
        assert user_model.preferred_depth == ResponseDepth.STANDARD

    @pytest.mark.asyncio
    async def test_load_user_model_caches(self, orchestrator):
        """Test user model is cached."""
        model1 = await orchestrator._load_user_model("test-user")
        model2 = await orchestrator._load_user_model("test-user")

        assert model1 is model2

    @pytest.mark.asyncio
    async def test_load_conversation_context_creates_new(self, orchestrator):
        """Test loading context creates new if not exists."""
        context = await orchestrator._load_conversation_context("session-1", "user-1")

        assert context.session_id == "session-1"
        assert context.user_id == "user-1"
        assert len(context.turns) == 0

    @pytest.mark.asyncio
    async def test_load_conversation_context_caches(self, orchestrator):
        """Test context is cached."""
        ctx1 = await orchestrator._load_conversation_context("session-1", "user-1")
        ctx2 = await orchestrator._load_conversation_context("session-1", "user-1")

        assert ctx1 is ctx2


class TestDeliberation:
    """Tests for deliberation parsing."""

    def test_parse_deliberation_valid_json(self, orchestrator):
        """Test parsing valid deliberation JSON."""
        response = '''
        {
            "understanding": "User wants to find papers on transformers",
            "can_answer_from_context": false,
            "knowledge_gaps": ["latest research"],
            "tools_to_use": [
                {"tool": "arxiv", "rationale": "Need academic papers", "query": "transformer architecture"}
            ],
            "action": "use_tools",
            "reasoning": "This is a research query requiring arxiv"
        }
        '''

        deliberation = orchestrator._parse_deliberation(response)

        assert deliberation.understanding == "User wants to find papers on transformers"
        assert not deliberation.can_answer_from_context
        assert deliberation.action == ActionDecision.USE_TOOLS
        assert len(deliberation.tools_to_use) == 1
        assert deliberation.tools_to_use[0].tool_name == "arxiv"

    def test_parse_deliberation_with_code_block(self, orchestrator):
        """Test parsing JSON in code block."""
        response = '''
        Here's my analysis:
        ```json
        {
            "understanding": "Follow-up about authors",
            "can_answer_from_context": true,
            "knowledge_gaps": [],
            "tools_to_use": [],
            "action": "respond_directly",
            "reasoning": "Already have this info"
        }
        ```
        '''

        deliberation = orchestrator._parse_deliberation(response)

        assert deliberation.can_answer_from_context
        assert deliberation.action == ActionDecision.RESPOND_DIRECTLY
        assert len(deliberation.tools_to_use) == 0

    def test_parse_deliberation_invalid_json(self, orchestrator):
        """Test parsing invalid JSON returns default."""
        response = "This is not JSON"

        deliberation = orchestrator._parse_deliberation(response)

        assert deliberation.action == ActionDecision.RESPOND_DIRECTLY
        assert "Parse error" in deliberation.reasoning


class TestUserModel:
    """Tests for UserModel."""

    def test_user_model_creation(self):
        """Test basic user model creation."""
        user = UserModel(user_id="test")

        assert user.user_id == "test"
        assert user.preferred_depth == ResponseDepth.STANDARD
        assert user.prefers_code_examples is True

    def test_user_model_expertise(self):
        """Test expertise methods."""
        user = UserModel(
            user_id="test",
            expertise_areas={"transformers": ExpertiseLevel.EXPERT}
        )

        assert user.get_expertise("transformers") == ExpertiseLevel.EXPERT
        assert user.get_expertise("unknown") == ExpertiseLevel.UNKNOWN

    def test_user_model_expertise_partial_match(self):
        """Test expertise with partial topic match."""
        user = UserModel(
            user_id="test",
            expertise_areas={"machine learning": ExpertiseLevel.INTERMEDIATE}
        )

        # Should match partial
        assert user.get_expertise("machine learning models") == ExpertiseLevel.INTERMEDIATE

    def test_user_model_serialization(self):
        """Test to_dict and from_dict."""
        user = UserModel(
            user_id="test",
            expertise_areas={"ml": ExpertiseLevel.EXPERT},
            preferred_depth=ResponseDepth.DETAILED,
        )

        data = user.to_dict()
        restored = UserModel.from_dict(data)

        assert restored.user_id == user.user_id
        assert restored.get_expertise("ml") == ExpertiseLevel.EXPERT
        assert restored.preferred_depth == ResponseDepth.DETAILED


class TestConversationContext:
    """Tests for ConversationContext."""

    def test_context_creation(self):
        """Test basic context creation."""
        ctx = ConversationContext(session_id="test")

        assert ctx.session_id == "test"
        assert len(ctx.turns) == 0
        assert ctx.current_topic == ""

    def test_add_turn(self):
        """Test adding turns."""
        ctx = ConversationContext(session_id="test")

        turn = ctx.add_turn("user", "Hello")

        assert len(ctx.turns) == 1
        assert turn.role == "user"
        assert turn.content == "Hello"

    def test_get_recent_turns(self):
        """Test getting recent turns."""
        ctx = ConversationContext(session_id="test")
        for i in range(10):
            ctx.add_turn("user", f"Message {i}")

        recent = ctx.get_recent_turns(3)

        assert len(recent) == 3
        assert recent[0].content == "Message 7"

    def test_add_paper(self):
        """Test adding paper to context."""
        ctx = ConversationContext(session_id="test")

        ctx.add_paper({
            "title": "Attention Is All You Need",
            "arxiv_id": "1706.03762"
        })

        assert len(ctx.papers_discussed) == 1
        assert ctx.papers_discussed[0]["title"] == "Attention Is All You Need"

    def test_add_paper_deduplicates(self):
        """Test adding duplicate paper updates instead of duplicating."""
        ctx = ConversationContext(session_id="test")

        ctx.add_paper({"title": "Paper", "arxiv_id": "123"})
        ctx.add_paper({"title": "Paper Updated", "arxiv_id": "123"})

        assert len(ctx.papers_discussed) == 1
        assert ctx.papers_discussed[0]["title"] == "Paper Updated"

    def test_get_entity_by_reference_numbered(self):
        """Test getting entity by numbered reference."""
        ctx = ConversationContext(session_id="test")
        ctx.add_paper({"title": "First Paper"})
        ctx.add_paper({"title": "Second Paper"})

        entity = ctx.get_entity_by_reference("the first paper")

        assert entity is not None
        assert entity["title"] == "First Paper"

    def test_update_topic(self):
        """Test topic tracking."""
        ctx = ConversationContext(session_id="test")

        ctx.update_topic("transformers")
        ctx.update_topic("attention mechanisms")

        assert ctx.current_topic == "attention mechanisms"
        assert "transformers" in ctx.topic_history

    def test_serialization(self):
        """Test to_dict and from_dict."""
        ctx = ConversationContext(session_id="test", user_id="user1")
        ctx.add_turn("user", "Hello")
        ctx.update_topic("ML")

        data = ctx.to_dict()
        restored = ConversationContext.from_dict(data)

        assert restored.session_id == ctx.session_id
        assert restored.user_id == ctx.user_id
        assert len(restored.turns) == 1
        assert restored.current_topic == "ML"


class TestStyleInstructions:
    """Tests for style personalization."""

    def test_get_style_instructions_brief(self, orchestrator):
        """Test brief style instructions."""
        user = UserModel(
            user_id="test",
            preferred_depth=ResponseDepth.BRIEF
        )

        instructions = orchestrator._get_style_instructions(user)

        assert "concise" in instructions.lower()

    def test_get_style_instructions_expert(self, orchestrator):
        """Test expert style instructions."""
        user = UserModel(
            user_id="test",
            expertise_areas={"ml": ExpertiseLevel.EXPERT}
        )

        instructions = orchestrator._get_style_instructions(user)

        assert "expert" in instructions.lower()

    def test_get_style_instructions_code_examples(self, orchestrator):
        """Test code examples preference."""
        user = UserModel(
            user_id="test",
            prefers_code_examples=True
        )

        instructions = orchestrator._get_style_instructions(user)

        assert "code" in instructions.lower()
