"""
Unit Tests for DOVA Collaborative Reasoning.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from dova.services.collaborative import (
    CollaborativeReasoning,
    CollaborativeResult,
    CollaborationMode,
    ToolExecutionResult,
)
from dova.services.blackboard import PostType
from dova.services.ensemble import AggregationMethod


@pytest.fixture
def mock_llm_func() -> AsyncMock:
    """Create mock LLM function."""
    async def llm_func(prompt: str) -> str:
        return f"Response to: {prompt[:50]}"
    return AsyncMock(side_effect=llm_func)


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock settings."""
    settings = MagicMock()
    settings.mcp_servers = ["arxiv", "github"]
    return settings


@pytest.fixture
def mock_mcp_client() -> AsyncMock:
    """Create mock MCP client."""
    client = AsyncMock()
    client.list_tools = AsyncMock(return_value=[
        {"name": "search", "description": "Search papers"},
    ])
    client.invoke = AsyncMock(return_value={"results": ["paper1", "paper2"]})
    return client


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create mock agent with reasoning capabilities."""
    agent = MagicMock()
    agent.name = "TestAgent"

    # Mock reason method
    async def mock_reason(problem, context=None, max_iterations=3):
        result = MagicMock()
        result.final_answer = f"Answer to: {problem[:30]}"
        result.refined_answer = None
        result.confidence = 0.8
        return result

    agent.reason = AsyncMock(side_effect=mock_reason)

    # Mock think method
    async def mock_think(prompt):
        return f"Thought: {prompt[:30]}"

    agent.think = AsyncMock(side_effect=mock_think)

    # Mock reflect method
    async def mock_reflect(answer, problem):
        return (f"Refined: {answer[:20]}", "Critique: looks good")

    agent.reflect = AsyncMock(side_effect=mock_reflect)

    return agent


@pytest.fixture
def collaborative_reasoning(
    mock_llm_func: AsyncMock,
    mock_settings: MagicMock,
    mock_mcp_client: AsyncMock,
) -> CollaborativeReasoning:
    """Create collaborative reasoning instance."""
    return CollaborativeReasoning(
        llm_func=mock_llm_func,
        settings=mock_settings,
        mcp_client=mock_mcp_client,
    )


class TestCollaborativeReasoning:
    """Test cases for CollaborativeReasoning."""

    def test_initialization(
        self,
        mock_llm_func: AsyncMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test collaborative reasoning initializes correctly."""
        collab = CollaborativeReasoning(
            llm_func=mock_llm_func,
            settings=mock_settings,
        )

        assert collab.llm_func == mock_llm_func
        assert collab.settings == mock_settings
        assert collab.blackboard is not None
        assert collab.ensemble is not None

    def test_collaboration_modes(self) -> None:
        """Test all collaboration modes are defined."""
        assert CollaborationMode.BLACKBOARD.value == "blackboard"
        assert CollaborationMode.ENSEMBLE.value == "ensemble"
        assert CollaborationMode.ITERATIVE.value == "iterative"
        assert CollaborationMode.HYBRID.value == "hybrid"
        assert CollaborationMode.TOOL_AUGMENTED.value == "tool_augmented"


class TestBlackboardReasoning:
    """Test cases for blackboard-based reasoning."""

    @pytest.mark.asyncio
    async def test_blackboard_reasoning_single_agent(
        self,
        collaborative_reasoning: CollaborativeReasoning,
        mock_agent: MagicMock,
    ) -> None:
        """Test blackboard reasoning with single agent."""
        result = await collaborative_reasoning.reason(
            problem="What is machine learning?",
            agents=[mock_agent],
            mode=CollaborationMode.BLACKBOARD,
            use_tools=False,
        )

        assert isinstance(result, CollaborativeResult)
        assert result.mode_used == CollaborationMode.BLACKBOARD
        assert "TestAgent" in result.participating_agents
        assert result.blackboard_synthesis is not None

    @pytest.mark.asyncio
    async def test_blackboard_reasoning_multiple_agents(
        self,
        collaborative_reasoning: CollaborativeReasoning,
        mock_agent: MagicMock,
    ) -> None:
        """Test blackboard reasoning with multiple agents."""
        agent2 = MagicMock()
        agent2.name = "Agent2"
        agent2.reason = mock_agent.reason
        agent2.think = mock_agent.think

        result = await collaborative_reasoning.reason(
            problem="Compare neural networks",
            agents=[mock_agent, agent2],
            mode=CollaborationMode.BLACKBOARD,
            use_tools=False,
        )

        assert len(result.participating_agents) >= 1
        assert result.blackboard_synthesis is not None


class TestEnsembleReasoning:
    """Test cases for ensemble-based reasoning."""

    @pytest.mark.asyncio
    async def test_ensemble_reasoning(
        self,
        collaborative_reasoning: CollaborativeReasoning,
        mock_agent: MagicMock,
    ) -> None:
        """Test ensemble reasoning mode."""
        result = await collaborative_reasoning.reason(
            problem="Explain transformers",
            agents=[mock_agent],
            mode=CollaborationMode.ENSEMBLE,
            use_tools=False,
        )

        assert isinstance(result, CollaborativeResult)
        assert result.mode_used == CollaborationMode.ENSEMBLE
        assert result.ensemble_result is not None


class TestIterativeReasoning:
    """Test cases for iterative refinement reasoning."""

    @pytest.mark.asyncio
    async def test_iterative_reasoning(
        self,
        collaborative_reasoning: CollaborativeReasoning,
        mock_agent: MagicMock,
    ) -> None:
        """Test iterative reasoning mode."""
        result = await collaborative_reasoning.reason(
            problem="Design a recommendation system",
            agents=[mock_agent],
            mode=CollaborationMode.ITERATIVE,
            max_iterations=2,
            use_tools=False,
        )

        assert isinstance(result, CollaborativeResult)
        assert result.mode_used == CollaborationMode.ITERATIVE
        assert result.iterations == 2
        assert len(result.refinement_history) > 0

    @pytest.mark.asyncio
    async def test_iterative_reasoning_empty_agents(
        self,
        collaborative_reasoning: CollaborativeReasoning,
    ) -> None:
        """Test iterative reasoning with no agents."""
        result = await collaborative_reasoning.reason(
            problem="Test problem",
            agents=[],
            mode=CollaborationMode.ITERATIVE,
            use_tools=False,
        )

        assert result.final_answer == ""
        assert result.confidence == 0.0


class TestHybridReasoning:
    """Test cases for hybrid reasoning."""

    @pytest.mark.asyncio
    async def test_hybrid_reasoning(
        self,
        collaborative_reasoning: CollaborativeReasoning,
        mock_agent: MagicMock,
    ) -> None:
        """Test hybrid reasoning mode."""
        result = await collaborative_reasoning.reason(
            problem="Analyze RAG vs fine-tuning",
            agents=[mock_agent],
            mode=CollaborationMode.HYBRID,
            max_iterations=2,
            use_tools=False,
        )

        assert isinstance(result, CollaborativeResult)
        assert result.mode_used == CollaborationMode.HYBRID
        assert result.blackboard_synthesis is not None
        assert result.ensemble_result is not None


class TestToolAugmentedReasoning:
    """Test cases for tool-augmented reasoning."""

    @pytest.mark.asyncio
    async def test_tool_augmented_reasoning(
        self,
        mock_llm_func: AsyncMock,
        mock_settings: MagicMock,
        mock_mcp_client: AsyncMock,
        mock_agent: MagicMock,
    ) -> None:
        """Test tool-augmented reasoning mode."""
        # Create collaborative reasoning with proper settings for tool resolver
        mock_settings.sandbox = MagicMock()
        mock_settings.sandbox.enabled = False
        mock_settings.aws = MagicMock()
        mock_settings.aws.agentcore_memory_enabled = False

        collab = CollaborativeReasoning(
            llm_func=mock_llm_func,
            settings=mock_settings,
            mcp_client=mock_mcp_client,
        )

        result = await collab.reason(
            problem="Find papers on attention mechanisms",
            agents=[mock_agent],
            mode=CollaborationMode.TOOL_AUGMENTED,
            max_iterations=2,
        )

        assert isinstance(result, CollaborativeResult)
        assert result.mode_used == CollaborationMode.TOOL_AUGMENTED
        assert result.tool_plan is not None

    @pytest.mark.asyncio
    async def test_tool_augmented_no_agents(
        self,
        mock_llm_func: AsyncMock,
        mock_settings: MagicMock,
        mock_mcp_client: AsyncMock,
    ) -> None:
        """Test tool-augmented reasoning without agents."""
        mock_settings.sandbox = MagicMock()
        mock_settings.sandbox.enabled = False
        mock_settings.aws = MagicMock()
        mock_settings.aws.agentcore_memory_enabled = False

        collab = CollaborativeReasoning(
            llm_func=mock_llm_func,
            settings=mock_settings,
            mcp_client=mock_mcp_client,
        )

        result = await collab.reason(
            problem="Search for transformers",
            agents=[],
            mode=CollaborationMode.TOOL_AUGMENTED,
        )

        assert isinstance(result, CollaborativeResult)
        assert result.mode_used == CollaborationMode.TOOL_AUGMENTED
        # Should still work, synthesizing from tool results
        assert result.final_answer is not None


class TestToolContextGathering:
    """Test cases for tool context gathering."""

    @pytest.mark.asyncio
    async def test_gather_tool_context(
        self,
        collaborative_reasoning: CollaborativeReasoning,
    ) -> None:
        """Test gathering tool context."""
        # This should not raise even without proper tool setup
        results = await collaborative_reasoning._gather_tool_context(
            "Find machine learning papers",
            context=None,
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_reasoning_with_tools_enabled(
        self,
        collaborative_reasoning: CollaborativeReasoning,
        mock_agent: MagicMock,
    ) -> None:
        """Test reasoning with tools enabled (default)."""
        result = await collaborative_reasoning.reason(
            problem="Research neural networks",
            agents=[mock_agent],
            mode=CollaborationMode.BLACKBOARD,
            use_tools=True,  # Default
        )

        assert isinstance(result, CollaborativeResult)

    @pytest.mark.asyncio
    async def test_reasoning_with_tools_disabled(
        self,
        collaborative_reasoning: CollaborativeReasoning,
        mock_agent: MagicMock,
    ) -> None:
        """Test reasoning with tools disabled."""
        result = await collaborative_reasoning.reason(
            problem="Research neural networks",
            agents=[mock_agent],
            mode=CollaborationMode.BLACKBOARD,
            use_tools=False,
        )

        assert isinstance(result, CollaborativeResult)
        assert len(result.tools_used) == 0


class TestToolExecutionResult:
    """Test cases for ToolExecutionResult dataclass."""

    def test_tool_execution_result_success(self) -> None:
        """Test successful tool execution result."""
        result = ToolExecutionResult(
            tool_name="mcp:arxiv:search",
            success=True,
            data={"papers": ["paper1", "paper2"]},
            execution_time_ms=150.5,
        )

        assert result.tool_name == "mcp:arxiv:search"
        assert result.success is True
        assert result.data is not None
        assert result.error is None

    def test_tool_execution_result_failure(self) -> None:
        """Test failed tool execution result."""
        result = ToolExecutionResult(
            tool_name="mcp:github:search",
            success=False,
            error="Connection timeout",
            execution_time_ms=5000.0,
        )

        assert result.success is False
        assert result.error == "Connection timeout"
        assert result.data is None


class TestCollaborativeResult:
    """Test cases for CollaborativeResult dataclass."""

    def test_collaborative_result_minimal(self) -> None:
        """Test minimal collaborative result."""
        result = CollaborativeResult(
            final_answer="Test answer",
            confidence=0.9,
            mode_used=CollaborationMode.BLACKBOARD,
        )

        assert result.final_answer == "Test answer"
        assert result.confidence == 0.9
        assert result.iterations == 0
        assert result.participating_agents == []
        assert result.tools_used == []

    def test_collaborative_result_full(self) -> None:
        """Test full collaborative result with all fields."""
        tool_result = ToolExecutionResult(
            tool_name="test_tool",
            success=True,
            data={"key": "value"},
        )

        result = CollaborativeResult(
            final_answer="Comprehensive answer",
            confidence=0.95,
            mode_used=CollaborationMode.TOOL_AUGMENTED,
            iterations=3,
            participating_agents=["Agent1", "Agent2"],
            tools_used=["tool1", "tool2"],
            tool_results=[tool_result],
        )

        assert len(result.participating_agents) == 2
        assert len(result.tools_used) == 2
        assert len(result.tool_results) == 1


class TestExtractQuery:
    """Test cases for query extraction."""

    def test_extract_query_simple(
        self,
        collaborative_reasoning: CollaborativeReasoning,
    ) -> None:
        """Test simple query extraction."""
        query = collaborative_reasoning._extract_query(
            "What are the latest advances in machine learning?"
        )

        assert isinstance(query, str)
        assert len(query) > 0
        # Should remove common question words
        assert "what" not in query.lower() or len(query) < 100

    def test_extract_query_long(
        self,
        collaborative_reasoning: CollaborativeReasoning,
    ) -> None:
        """Test query extraction from long text."""
        long_problem = "a" * 200
        query = collaborative_reasoning._extract_query(long_problem)

        # Should be truncated
        assert len(query) <= 100

    def test_extract_query_with_sentence(
        self,
        collaborative_reasoning: CollaborativeReasoning,
    ) -> None:
        """Test query extraction stops at sentence boundary."""
        problem = "Find papers about transformers. Also include GPT models."
        query = collaborative_reasoning._extract_query(problem)

        # Should stop at first sentence
        assert "Also" not in query


class TestSynthesizeToolResults:
    """Test cases for tool result synthesis."""

    def test_synthesize_empty_results(
        self,
        collaborative_reasoning: CollaborativeReasoning,
    ) -> None:
        """Test synthesis with no results."""
        answer = collaborative_reasoning._synthesize_tool_results(
            "test problem",
            [],
        )

        assert "Unable to find" in answer

    def test_synthesize_failed_results(
        self,
        collaborative_reasoning: CollaborativeReasoning,
    ) -> None:
        """Test synthesis with failed results only."""
        results = [
            ToolExecutionResult(
                tool_name="tool1",
                success=False,
                error="Failed",
            ),
        ]

        answer = collaborative_reasoning._synthesize_tool_results(
            "test problem",
            results,
        )

        assert "no useful data" in answer

    def test_synthesize_successful_results(
        self,
        collaborative_reasoning: CollaborativeReasoning,
    ) -> None:
        """Test synthesis with successful results."""
        results = [
            ToolExecutionResult(
                tool_name="arxiv_search",
                success=True,
                data={"summary": "Found 5 relevant papers"},
            ),
            ToolExecutionResult(
                tool_name="github_search",
                success=True,
                data=["repo1", "repo2", "repo3"],
            ),
        ]

        answer = collaborative_reasoning._synthesize_tool_results(
            "find ML resources",
            results,
        )

        assert "Based on analysis" in answer
        assert "arxiv_search" in answer
        assert "github_search" in answer
