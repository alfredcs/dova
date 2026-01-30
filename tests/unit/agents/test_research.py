"""
Unit Tests for DOVA Research Agent.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from dova.agents.base import AgentResult, AgentTask
from dova.agents.research import ResearchAgent, ResearchFindings, SearchResult
from dova.config.providers import LLMRouter


@pytest.fixture
def research_agent(mock_llm_router: LLMRouter, mock_mcp_client: AsyncMock) -> ResearchAgent:
    """Create research agent instance for testing."""
    return ResearchAgent(
        llm_router=mock_llm_router,
        mcp_client=mock_mcp_client,
    )


class TestResearchAgent:
    """Test cases for ResearchAgent."""

    def test_research_agent_initialization(
        self, mock_llm_router: LLMRouter, mock_mcp_client: AsyncMock
    ) -> None:
        """Test research agent initializes correctly."""
        agent = ResearchAgent(
            llm_router=mock_llm_router,
            mcp_client=mock_mcp_client,
        )

        assert agent.name == "ResearchAgent"
        assert agent.llm_router == mock_llm_router

    @pytest.mark.asyncio
    async def test_execute_without_query(self, research_agent: ResearchAgent) -> None:
        """Test execute fails without query."""
        task = AgentTask(
            type="research",
            params={},  # No query
            user_id="test-user",
        )

        result = await research_agent.execute(task)

        assert not result.success
        assert "query" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_query(self, research_agent: ResearchAgent) -> None:
        """Test execute with valid query."""
        task = AgentTask(
            type="research",
            params={"query": "machine learning transformers", "source": "arxiv"},
            user_id="test-user",
        )

        result = await research_agent.execute(task)

        assert result.success
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_search_arxiv(self, research_agent: ResearchAgent) -> None:
        """Test ArXiv search."""
        results = await research_agent._search_arxiv("neural networks", {})

        assert isinstance(results, ResearchFindings)

    @pytest.mark.asyncio
    async def test_search_github(self, research_agent: ResearchAgent) -> None:
        """Test GitHub search."""
        results = await research_agent._search_github("pytorch transformers", {})

        assert isinstance(results, ResearchFindings)

    @pytest.mark.asyncio
    async def test_search_huggingface(self, research_agent: ResearchAgent) -> None:
        """Test HuggingFace search."""
        results = await research_agent._search_huggingface("bert model", {})

        assert isinstance(results, ResearchFindings)

    @pytest.mark.asyncio
    async def test_execute_with_sources_filter(
        self, research_agent: ResearchAgent
    ) -> None:
        """Test execute with specific source."""
        task = AgentTask(
            type="research",
            params={
                "query": "deep learning",
                "source": "github",
            },
            user_id="test-user",
        )

        result = await research_agent.execute(task)

        assert result.success

    @pytest.mark.asyncio
    async def test_mcp_client_error_handling(
        self, research_agent: ResearchAgent
    ) -> None:
        """Test handling of MCP client errors."""
        # Make MCP client raise an error
        research_agent.mcp_client.invoke = AsyncMock(
            side_effect=Exception("MCP connection failed")
        )

        task = AgentTask(
            type="research",
            params={"query": "test query", "source": "arxiv"},
            user_id="test-user",
        )

        result = await research_agent.execute(task)

        # Should handle error gracefully
        assert isinstance(result, AgentResult)


class TestSearchResult:
    """Test cases for SearchResult model."""

    def test_search_result_creation(self) -> None:
        """Test creating search result."""
        result = SearchResult(
            source="arxiv",
            title="Test Paper",
            url="https://arxiv.org/abs/1234.5678",
            description="Test description",
            metadata={"authors": ["Author 1"]},
            relevance_score=0.9,
        )

        assert result.source == "arxiv"
        assert result.title == "Test Paper"
        assert result.relevance_score == 0.9

    def test_search_result_defaults(self) -> None:
        """Test search result default values."""
        result = SearchResult(
            source="github",
            title="Test Repo",
            url="https://github.com/test/repo",
        )

        assert result.description == ""
        assert result.metadata == {}
        assert result.relevance_score == 0.0


class TestResearchFindings:
    """Test cases for ResearchFindings model."""

    def test_research_findings_creation(self) -> None:
        """Test creating research findings."""
        paper = SearchResult(
            source="arxiv",
            title="Test Paper",
            url="https://arxiv.org/abs/1234",
        )
        repo = SearchResult(
            source="github",
            title="test/repo",
            url="https://github.com/test/repo",
        )

        findings = ResearchFindings(
            papers=[paper],
            repositories=[repo],
        )

        assert len(findings.papers) == 1
        assert len(findings.repositories) == 1
        assert len(findings.models) == 0

    def test_research_findings_defaults(self) -> None:
        """Test research findings default values."""
        findings = ResearchFindings()

        assert findings.papers == []
        assert findings.repositories == []
        assert findings.models == []
        assert findings.datasets == []
        assert findings.web_results == []
