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


class TestDebateFirstFlow:
    """Tests for the debate-before-synthesis architecture.

    The debate runs on research evidence BEFORE synthesis, and the debate's
    bull/bear output is injected into the synthesis prompt so the final
    answer reflects both sides. Low-confidence debates trigger one extra
    refinement pass that explicitly addresses bear concerns.
    """

    @pytest.fixture
    def debate_out_high_conf(self):
        return {
            "bull_strengths": ["strong community", "fast inference"],
            "bear_concerns": ["limited multilingual support"],
            "balanced_assessment": "Net positive for English-only use cases.",
            "recommendation": "Adopt with English-only caveat.",
            "confidence_score": 0.85,
            "debate_summary": "Strong bull case, minor bear concerns.",
        }

    @pytest.fixture
    def debate_out_low_conf(self):
        return {
            "bull_strengths": ["permissive license"],
            "bear_concerns": [
                "unproven at scale",
                "limited benchmark coverage",
            ],
            "balanced_assessment": "Evidence is thin on both sides.",
            "recommendation": "Pilot before adopting.",
            "confidence_score": 0.45,
            "debate_summary": "Low-confidence outcome.",
        }

    @pytest.fixture
    def sample_results(self):
        return {
            "papers": [{"title": "A paper", "url": "https://arxiv.org/abs/1", "description": "x"}],
            "repositories": [],
            "models": [],
            "web_results": [],
        }

    @pytest.fixture
    def default_deliberation(self):
        return Deliberation(
            understanding="compare X and Y",
            can_answer_from_context=False,
            action=ActionDecision.USE_TOOLS,
            tools_to_use=[
                ToolConsideration(
                    tool_name="arxiv", would_help=True,
                    rationale="search", search_query="q",
                )
            ],
            intent_weights={"ai": 0.7, "bio": 0.1, "web": 0.2},
        )

    def test_build_prompt_without_debate_omits_block(
        self, orchestrator, sample_results, default_deliberation,
    ):
        """No debate output → synthesis prompt has no Adversarial Analysis."""
        user = UserModel(user_id="u")
        context = ConversationContext(session_id="s", user_id="u")
        prompt = orchestrator._build_synthesis_prompt(
            "q", sample_results, user, context, default_deliberation,
            debate_output=None,
        )
        assert "Adversarial Analysis" not in prompt
        assert "Bull strengths" not in prompt
        assert "Bear concerns" not in prompt

    def test_build_prompt_with_debate_injects_block(
        self, orchestrator, sample_results, default_deliberation,
        debate_out_high_conf,
    ):
        """Debate output → synthesis prompt includes bull/bear sections."""
        user = UserModel(user_id="u")
        context = ConversationContext(session_id="s", user_id="u")
        prompt = orchestrator._build_synthesis_prompt(
            "q", sample_results, user, context, default_deliberation,
            debate_output=debate_out_high_conf,
        )
        assert "Adversarial Analysis" in prompt
        assert "strong community" in prompt
        assert "limited multilingual support" in prompt
        assert "Moderator recommendation" in prompt
        assert "DEBATE INTEGRATION RULES" in prompt
        # Synthesis LLM must be told to address every bear concern.
        assert "address EVERY concern" in prompt

    @pytest.mark.asyncio
    async def test_debate_runs_before_synthesis(
        self, orchestrator, sample_results, debate_out_high_conf,
    ):
        """End-to-end call order: research → debate → synthesis (not the other way)."""
        call_order: list[str] = []

        async def fake_tools(*_a, **_kw):
            call_order.append("research")
            return sample_results

        async def fake_run_debate(*_a, **_kw):
            call_order.append("debate")
            return debate_out_high_conf

        async def fake_synth(_query, _results, _user, _ctx, _delib, debate_output=None):
            call_order.append("synthesis")
            assert debate_output is debate_out_high_conf, (
                "synthesis must receive debate_output so the final answer "
                "reflects bull/bear conclusions"
            )
            return "final answer"

        orchestrator._execute_selected_tools = AsyncMock(side_effect=fake_tools)
        orchestrator._run_debate = AsyncMock(side_effect=fake_run_debate)
        orchestrator._synthesize_with_results = AsyncMock(side_effect=fake_synth)
        orchestrator._deliberate = AsyncMock(
            return_value=Deliberation(
                understanding="compare",
                can_answer_from_context=False,
                action=ActionDecision.USE_TOOLS,
                tools_to_use=[
                    ToolConsideration(
                        tool_name="arxiv", would_help=True,
                        rationale="r", search_query="q",
                    )
                ],
                intent_weights={"ai": 1.0, "bio": 0.0, "web": 0.0},
            )
        )
        orchestrator.agents["debate"] = MagicMock()

        task = AgentTask(
            type="query",
            user_id="u",
            params={"query": "compare X and Y", "auto_debate": True},
        )
        result = await orchestrator.execute(task)

        assert result.success
        assert call_order == ["research", "debate", "synthesis"], (
            f"Expected research → debate → synthesis, got {call_order}"
        )

    @pytest.mark.asyncio
    async def test_no_debate_preserves_single_synthesis_path(
        self, orchestrator, sample_results,
    ):
        """When debate disabled, synthesis is called once with debate_output=None."""
        call_order: list[str] = []

        async def fake_tools(*_a, **_kw):
            call_order.append("research")
            return sample_results

        async def fake_synth(_query, _results, _user, _ctx, _delib, debate_output=None):
            call_order.append("synthesis")
            assert debate_output is None
            return "final"

        orchestrator._execute_selected_tools = AsyncMock(side_effect=fake_tools)
        orchestrator._run_debate = AsyncMock(
            side_effect=AssertionError("debate must not run when auto_debate=False")
        )
        orchestrator._synthesize_with_results = AsyncMock(side_effect=fake_synth)
        orchestrator._deliberate = AsyncMock(
            return_value=Deliberation(
                understanding="q",
                can_answer_from_context=False,
                action=ActionDecision.USE_TOOLS,
                tools_to_use=[
                    ToolConsideration(
                        tool_name="arxiv", would_help=True,
                        rationale="r", search_query="q",
                    )
                ],
                intent_weights={"ai": 1.0, "bio": 0.0, "web": 0.0},
            )
        )
        orchestrator.agents["debate"] = MagicMock()

        task = AgentTask(
            type="query", user_id="u",
            params={"query": "what is a transformer", "auto_debate": False},
        )
        result = await orchestrator.execute(task)

        assert result.success
        assert call_order == ["research", "synthesis"]

    @pytest.mark.asyncio
    async def test_non_evaluative_query_skips_debate(
        self, orchestrator, sample_results,
    ):
        """auto_debate=True + non-evaluative query → debate still skipped."""
        synth_calls = []
        orchestrator._execute_selected_tools = AsyncMock(return_value=sample_results)
        orchestrator._run_debate = AsyncMock(
            side_effect=AssertionError("should not debate non-evaluative queries")
        )

        async def fake_synth(_query, _results, _user, _ctx, _delib, debate_output=None):
            synth_calls.append(debate_output)
            return "final"

        orchestrator._synthesize_with_results = AsyncMock(side_effect=fake_synth)
        orchestrator._deliberate = AsyncMock(
            return_value=Deliberation(
                understanding="q",
                can_answer_from_context=False,
                action=ActionDecision.USE_TOOLS,
                tools_to_use=[
                    ToolConsideration(
                        tool_name="arxiv", would_help=True,
                        rationale="r", search_query="q",
                    )
                ],
                intent_weights={"ai": 1.0, "bio": 0.0, "web": 0.0},
            )
        )
        orchestrator.agents["debate"] = MagicMock()

        task = AgentTask(
            type="query", user_id="u",
            # "explain" is not an evaluative pattern
            params={"query": "explain transformers", "auto_debate": True},
        )
        result = await orchestrator.execute(task)
        assert result.success
        assert synth_calls == [None]

    @pytest.mark.asyncio
    async def test_force_debate_overrides_evaluative_check(
        self, orchestrator, sample_results, debate_out_high_conf,
    ):
        """force_debate=True runs debate even for non-evaluative queries."""
        debate_ran = []

        async def fake_run_debate(*_a, **_kw):
            debate_ran.append(True)
            return debate_out_high_conf

        orchestrator._execute_selected_tools = AsyncMock(return_value=sample_results)
        orchestrator._run_debate = AsyncMock(side_effect=fake_run_debate)
        orchestrator._synthesize_with_results = AsyncMock(return_value="final")
        orchestrator._deliberate = AsyncMock(
            return_value=Deliberation(
                understanding="q",
                can_answer_from_context=False,
                action=ActionDecision.USE_TOOLS,
                tools_to_use=[
                    ToolConsideration(
                        tool_name="arxiv", would_help=True,
                        rationale="r", search_query="q",
                    )
                ],
                intent_weights={"ai": 1.0, "bio": 0.0, "web": 0.0},
            )
        )
        orchestrator.agents["debate"] = MagicMock()

        task = AgentTask(
            type="query", user_id="u",
            params={
                "query": "explain transformers",  # not evaluative
                "auto_debate": False,
                "force_debate": True,
            },
        )
        await orchestrator.execute(task)
        assert debate_ran == [True]

    @pytest.mark.asyncio
    async def test_refinement_triggers_on_low_confidence(
        self, orchestrator, sample_results, debate_out_low_conf,
    ):
        """Debate confidence < threshold → _refine_synthesis runs once."""
        orchestrator._execute_selected_tools = AsyncMock(return_value=sample_results)
        orchestrator._run_debate = AsyncMock(return_value=debate_out_low_conf)
        orchestrator._synthesize_with_results = AsyncMock(return_value="first draft")
        orchestrator._refine_synthesis = AsyncMock(return_value="refined answer")
        orchestrator._deliberate = AsyncMock(
            return_value=Deliberation(
                understanding="compare",
                can_answer_from_context=False,
                action=ActionDecision.USE_TOOLS,
                tools_to_use=[
                    ToolConsideration(
                        tool_name="arxiv", would_help=True,
                        rationale="r", search_query="q",
                    )
                ],
                intent_weights={"ai": 1.0, "bio": 0.0, "web": 0.0},
            )
        )
        orchestrator.agents["debate"] = MagicMock()

        task = AgentTask(
            type="query", user_id="u",
            params={
                "query": "compare X vs Y",
                "auto_debate": True,
                "refine_threshold": 0.7,
            },
        )
        result = await orchestrator.execute(task)

        assert result.success
        assert result.data["response"] == "refined answer"
        orchestrator._refine_synthesis.assert_called_once()
        # action_result should flag the refinement
        assert result.data["action_result"]["refined"] is True
        assert "0.45" in result.data["action_result"]["refine_reason"]

    @pytest.mark.asyncio
    async def test_high_confidence_skips_refinement(
        self, orchestrator, sample_results, debate_out_high_conf,
    ):
        """Debate confidence >= threshold → no refinement, original synthesis wins."""
        orchestrator._execute_selected_tools = AsyncMock(return_value=sample_results)
        orchestrator._run_debate = AsyncMock(return_value=debate_out_high_conf)
        orchestrator._synthesize_with_results = AsyncMock(return_value="confident answer")
        orchestrator._refine_synthesis = AsyncMock(
            side_effect=AssertionError("refinement must not run at high confidence")
        )
        orchestrator._deliberate = AsyncMock(
            return_value=Deliberation(
                understanding="compare",
                can_answer_from_context=False,
                action=ActionDecision.USE_TOOLS,
                tools_to_use=[
                    ToolConsideration(
                        tool_name="arxiv", would_help=True,
                        rationale="r", search_query="q",
                    )
                ],
                intent_weights={"ai": 1.0, "bio": 0.0, "web": 0.0},
            )
        )
        orchestrator.agents["debate"] = MagicMock()

        task = AgentTask(
            type="query", user_id="u",
            params={"query": "compare X vs Y", "auto_debate": True},
        )
        result = await orchestrator.execute(task)

        assert result.success
        assert result.data["response"] == "confident answer"
        assert result.data["action_result"].get("refined") is None

    @pytest.mark.asyncio
    async def test_refinement_can_be_disabled(
        self, orchestrator, sample_results, debate_out_low_conf,
    ):
        """refine=False in task.params disables refinement even on low confidence."""
        orchestrator._execute_selected_tools = AsyncMock(return_value=sample_results)
        orchestrator._run_debate = AsyncMock(return_value=debate_out_low_conf)
        orchestrator._synthesize_with_results = AsyncMock(return_value="draft only")
        orchestrator._refine_synthesis = AsyncMock(
            side_effect=AssertionError("refinement must be disabled")
        )
        orchestrator._deliberate = AsyncMock(
            return_value=Deliberation(
                understanding="compare",
                can_answer_from_context=False,
                action=ActionDecision.USE_TOOLS,
                tools_to_use=[
                    ToolConsideration(
                        tool_name="arxiv", would_help=True,
                        rationale="r", search_query="q",
                    )
                ],
                intent_weights={"ai": 1.0, "bio": 0.0, "web": 0.0},
            )
        )
        orchestrator.agents["debate"] = MagicMock()

        task = AgentTask(
            type="query", user_id="u",
            params={
                "query": "compare X vs Y",
                "auto_debate": True,
                "refine": False,
            },
        )
        result = await orchestrator.execute(task)
        assert result.success
        assert result.data["response"] == "draft only"

    @pytest.mark.asyncio
    async def test_run_debate_accepts_no_synthesized_answer(
        self, orchestrator, sample_results, debate_out_high_conf,
    ):
        """_run_debate works with synthesized_answer=None (pre-synthesis path)."""
        # Capture the context the debate agent receives.
        captured: dict = {}

        mock_debate_agent = MagicMock()

        async def capture_execute(agent_task):
            captured["params"] = agent_task.params
            # Return a successful AgentResult with debate-shaped data.
            r = MagicMock()
            r.success = True
            r.data = debate_out_high_conf
            return r

        mock_debate_agent.execute = AsyncMock(side_effect=capture_execute)
        orchestrator.agents["debate"] = mock_debate_agent

        deliberation = Deliberation(
            understanding="compare",
            can_answer_from_context=False,
            action=ActionDecision.USE_TOOLS,
        )
        out = await orchestrator._run_debate(
            "compare X vs Y", deliberation, sample_results,
            synthesized_answer=None,
        )
        assert out is not None
        # Without a synthesized answer, the context must NOT include orchestrator_answer
        assert "orchestrator_answer" not in captured["params"]["context"]
        # It SHOULD include the evidence
        assert "papers" in captured["params"]["context"]
