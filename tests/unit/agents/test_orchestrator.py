"""
Unit Tests for DOVA Orchestrator Agent.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from dova.agents.base import AgentResult, AgentTask
from dova.agents.orchestrator import DOVAOrchestrator, UserIntent, ParsedIntent
from dova.config.providers import LLMRouter


@pytest.fixture
def orchestrator(mock_llm_router: LLMRouter, mock_mcp_client: AsyncMock) -> DOVAOrchestrator:
    """Create orchestrator instance for testing."""
    return DOVAOrchestrator(
        llm_router=mock_llm_router,
        mcp_client=mock_mcp_client,
    )


class TestDOVAOrchestrator:
    """Test cases for DOVAOrchestrator."""

    def test_orchestrator_initialization(
        self, mock_llm_router: LLMRouter, mock_mcp_client: AsyncMock
    ) -> None:
        """Test orchestrator initializes correctly."""
        orchestrator = DOVAOrchestrator(
            llm_router=mock_llm_router,
            mcp_client=mock_mcp_client,
        )

        assert orchestrator.name == "DOVAOrchestrator"
        assert orchestrator.llm_router == mock_llm_router
        assert orchestrator.agents == {}

    def test_register_agent(
        self, orchestrator: DOVAOrchestrator, mock_llm_router: LLMRouter
    ) -> None:
        """Test agent registration."""
        from dova.agents.research import ResearchAgent

        research_agent = ResearchAgent(llm_router=mock_llm_router)
        orchestrator.register_agent("research", research_agent)

        assert "research" in orchestrator.agents
        assert orchestrator.agents["research"] == research_agent

    @pytest.mark.asyncio
    async def test_execute_without_query(self, orchestrator: DOVAOrchestrator) -> None:
        """Test execute fails without query."""
        task = AgentTask(
            type="research",
            params={},  # No query
            user_id="test-user",
        )

        result = await orchestrator.execute(task)

        assert not result.success
        assert "No query provided" in result.error

    @pytest.mark.asyncio
    async def test_classify_intent(self, orchestrator: DOVAOrchestrator) -> None:
        """Test intent classification."""
        # Mock the LLM response for classification
        orchestrator.llm_router.complete = AsyncMock(
            return_value=MagicMock(
                content='{"intent": "research_query", "confidence": 0.9, "entities": {"topics": ["machine learning"]}, "requires_profiling": false, "requires_validation": false}'
            )
        )

        intent = await orchestrator._classify_intent("latest advances in machine learning")

        assert intent.intent == UserIntent.RESEARCH_QUERY
        assert intent.confidence > 0.5
        assert "machine learning" in intent.entities.get("topics", [])

    @pytest.mark.asyncio
    async def test_build_task_graph(self, orchestrator: DOVAOrchestrator) -> None:
        """Test task graph building."""
        intent = ParsedIntent(
            intent=UserIntent.RESEARCH_QUERY,
            confidence=0.9,
            entities={"topics": ["NLP"]},
        )
        parent_task = AgentTask(
            type="research",
            params={"query": "test"},
            user_id="test-user",
        )

        graph = await orchestrator._build_task_graph("test query", intent, parent_task)

        # Should have search tasks for each source
        assert "arxiv_search" in graph
        assert "github_search" in graph
        assert "hf_search" in graph
        assert "synthesis" in graph

        # Synthesis should depend on searches
        assert "arxiv_search" in graph["synthesis"].dependencies
        assert "github_search" in graph["synthesis"].dependencies
        assert "hf_search" in graph["synthesis"].dependencies

    @pytest.mark.asyncio
    async def test_execute_task_graph_empty(self, orchestrator: DOVAOrchestrator) -> None:
        """Test executing empty task graph."""
        results = await orchestrator._execute_task_graph({})
        assert results == {}

    @pytest.mark.asyncio
    async def test_synthesize_results(self, orchestrator: DOVAOrchestrator) -> None:
        """Test result synthesis."""
        # Mock LLM response
        orchestrator.llm_router.complete = AsyncMock(
            return_value=MagicMock(
                content='{"summary": "Test synthesis", "papers": [], "code": [], "models": [], "insights": [], "recommendations": []}'
            )
        )

        results = {
            "arxiv_search": AgentResult(success=True, data={"papers": []}),
            "github_search": AgentResult(success=True, data={"repositories": []}),
        }

        intent = ParsedIntent(
            intent=UserIntent.RESEARCH_QUERY,
            confidence=0.9,
            entities={},
        )

        synthesis = await orchestrator._synthesize_results("test query", intent, results)

        assert "summary" in synthesis

    @pytest.mark.asyncio
    async def test_format_results_for_synthesis(
        self, orchestrator: DOVAOrchestrator
    ) -> None:
        """Test formatting results for synthesis prompt."""
        results = {
            "arxiv_search": AgentResult(
                success=True,
                data={
                    "papers": [
                        {
                            "title": "Test Paper",
                            "description": "Test description",
                        }
                    ],
                    "repositories": [],
                    "models": [],
                },
            ),
        }

        formatted = orchestrator._format_results_for_synthesis(results)

        assert "arxiv_search" in formatted
        assert "Test Paper" in formatted
