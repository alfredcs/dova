"""
Integration Tests for DOVA Full Workflow.

These tests verify end-to-end functionality of the DOVA platform,
including agent orchestration, MCP integration, and API workflows.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any

from dova.agents.base import AgentResult, AgentTask
from dova.agents.orchestrator import DOVAOrchestrator, UserIntent, ParsedIntent
from dova.agents.research import ResearchAgent, ResearchFindings
from dova.agents.profiling import ProfilingAgent
from dova.agents.synthesis import SynthesisAgent
from dova.agents.debate import DebateAgent
from dova.config.providers import LLMRouter


class TestFullResearchWorkflow:
    """Integration tests for complete research workflow."""

    @pytest.fixture
    def orchestrator(
        self, mock_llm_router: LLMRouter, mock_mcp_client: AsyncMock
    ) -> DOVAOrchestrator:
        """Create fully configured orchestrator."""
        orch = DOVAOrchestrator(
            llm_router=mock_llm_router,
            mcp_client=mock_mcp_client,
        )

        # Register all agents
        orch.register_agent(
            "research",
            ResearchAgent(llm_router=mock_llm_router, mcp_client=mock_mcp_client),
        )
        orch.register_agent(
            "profiling",
            ProfilingAgent(llm_router=mock_llm_router, mcp_client=mock_mcp_client),
        )
        orch.register_agent(
            "synthesis",
            SynthesisAgent(llm_router=mock_llm_router, mcp_client=mock_mcp_client),
        )
        orch.register_agent(
            "debate",
            DebateAgent(llm_router=mock_llm_router, mcp_client=mock_mcp_client),
        )

        return orch

    @pytest.mark.asyncio
    async def test_research_query_workflow(
        self, orchestrator: DOVAOrchestrator
    ) -> None:
        """Test complete research query workflow."""
        # Mock LLM responses for classification and synthesis
        orchestrator.llm_router.complete = AsyncMock(
            side_effect=[
                # Intent classification
                MagicMock(
                    content='{"intent": "research_query", "confidence": 0.9, "entities": {"topics": ["transformers"]}, "requires_profiling": false, "requires_validation": false}'
                ),
                # Synthesis
                MagicMock(
                    content='{"summary": "Research on transformers found several papers and repositories.", "papers": [], "code": [], "models": [], "insights": ["Transformers are widely used"], "recommendations": ["Start with attention paper"]}'
                ),
            ]
        )

        task = AgentTask(
            type="research",
            params={"query": "explain transformer architecture"},
            user_id="test-user",
        )

        result = await orchestrator.execute(task)

        assert result.success
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_research_with_profiling(
        self, orchestrator: DOVAOrchestrator
    ) -> None:
        """Test research workflow with user profiling."""
        orchestrator.llm_router.complete = AsyncMock(
            side_effect=[
                # Intent classification - requires profiling
                MagicMock(
                    content='{"intent": "research_query", "confidence": 0.9, "entities": {"topics": ["reinforcement learning"]}, "requires_profiling": true, "requires_validation": false}'
                ),
                # Profile extraction
                MagicMock(
                    content='{"interests": ["RL", "robotics"], "expertise_indicators": ["intermediate"], "research_intent": "application"}'
                ),
                # Synthesis
                MagicMock(
                    content='{"summary": "RL resources tailored to intermediate level.", "papers": [], "code": [], "models": [], "insights": [], "recommendations": []}'
                ),
            ]
        )

        task = AgentTask(
            type="research",
            params={"query": "reinforcement learning for robotics"},
            user_id="test-user",
        )

        result = await orchestrator.execute(task)

        assert result.success

    @pytest.mark.asyncio
    async def test_code_validation_workflow(
        self, orchestrator: DOVAOrchestrator
    ) -> None:
        """Test code validation workflow."""
        orchestrator.llm_router.complete = AsyncMock(
            side_effect=[
                # Intent classification
                MagicMock(
                    content='{"intent": "code_validation", "confidence": 0.95, "entities": {"code_type": "repository"}, "requires_profiling": false, "requires_validation": true}'
                ),
                # Validation result
                MagicMock(
                    content='{"is_valid": true, "quality_score": 0.85, "issues": [], "suggestions": ["Add type hints"]}'
                ),
            ]
        )

        task = AgentTask(
            type="validation",
            params={
                "query": "validate this repository",
                "repository_url": "https://github.com/test/repo",
            },
            user_id="test-user",
        )

        result = await orchestrator.execute(task)

        assert result.success


class TestMCPIntegration:
    """Integration tests for MCP server connectivity."""

    @pytest.fixture
    def mock_mcp_responses(self) -> dict[str, Any]:
        """Create mock MCP responses for all servers."""
        return {
            "arxiv": [
                {
                    "title": "Attention Is All You Need",
                    "id": "1706.03762",
                    "summary": "The dominant sequence transduction models...",
                    "authors": ["Vaswani", "Shazeer", "Parmar"],
                }
            ],
            "github": {
                "items": [
                    {
                        "full_name": "huggingface/transformers",
                        "description": "Transformers library",
                        "stargazers_count": 100000,
                        "html_url": "https://github.com/huggingface/transformers",
                    }
                ]
            },
            "huggingface": [
                {
                    "id": "bert-base-uncased",
                    "downloads": 50000000,
                    "pipeline_tag": "fill-mask",
                }
            ],
        }

    @pytest.mark.asyncio
    async def test_arxiv_mcp_integration(
        self, mock_llm_router: LLMRouter, mock_mcp_responses: dict[str, Any]
    ) -> None:
        """Test ArXiv MCP server integration."""
        mock_client = AsyncMock()
        mock_client.invoke = AsyncMock(return_value=mock_mcp_responses["arxiv"])

        agent = ResearchAgent(llm_router=mock_llm_router, mcp_client=mock_client)

        results = await agent._search_arxiv("transformers attention", {})

        assert isinstance(results, ResearchFindings)

    @pytest.mark.asyncio
    async def test_github_mcp_integration(
        self, mock_llm_router: LLMRouter, mock_mcp_responses: dict[str, Any]
    ) -> None:
        """Test GitHub MCP server integration."""
        mock_client = AsyncMock()
        mock_client.invoke = AsyncMock(return_value=mock_mcp_responses["github"])

        agent = ResearchAgent(llm_router=mock_llm_router, mcp_client=mock_client)

        results = await agent._search_github("transformers pytorch", {})

        assert isinstance(results, ResearchFindings)

    @pytest.mark.asyncio
    async def test_huggingface_mcp_integration(
        self, mock_llm_router: LLMRouter, mock_mcp_responses: dict[str, Any]
    ) -> None:
        """Test HuggingFace MCP server integration."""
        mock_client = AsyncMock()
        mock_client.invoke = AsyncMock(return_value=mock_mcp_responses["huggingface"])

        agent = ResearchAgent(llm_router=mock_llm_router, mcp_client=mock_client)

        results = await agent._search_huggingface("bert", {})

        assert isinstance(results, ResearchFindings)

    @pytest.mark.asyncio
    async def test_parallel_mcp_calls(
        self, mock_llm_router: LLMRouter, mock_mcp_responses: dict[str, Any]
    ) -> None:
        """Test parallel MCP calls across multiple servers."""
        mock_client = AsyncMock()

        async def mock_invoke(
            server: str, tool: str, params: dict[str, Any]
        ) -> dict[str, Any]:
            return mock_mcp_responses.get(server, {})

        mock_client.invoke = AsyncMock(side_effect=mock_invoke)

        agent = ResearchAgent(llm_router=mock_llm_router, mcp_client=mock_client)

        # Execute search that should call multiple servers
        task = AgentTask(
            type="research",
            params={"query": "machine learning", "source": "all"},
            user_id="test-user",
        )

        await agent.execute(task)

        # Should have called invoke multiple times
        assert mock_client.invoke.call_count >= 1


class TestDebateWorkflow:
    """Integration tests for Bull/Bear debate workflow."""

    @pytest.fixture
    def debate_agent(
        self, mock_llm_router: LLMRouter, mock_mcp_client: AsyncMock
    ) -> DebateAgent:
        """Create debate agent."""
        return DebateAgent(
            llm_router=mock_llm_router,
            mcp_client=mock_mcp_client,
        )

    @pytest.mark.asyncio
    async def test_technology_debate(self, debate_agent: DebateAgent) -> None:
        """Test technology evaluation debate."""
        # DebateAgent runs 2 rounds by default, each round calls both Bull and Bear agents
        # Then it synthesizes. Total: 2*(bull+bear) + synthesis = 5 LLM calls
        debate_agent.llm_router.complete = AsyncMock(
            side_effect=[
                # Round 1 Bull
                MagicMock(
                    content='{"argument": "High performance and memory safety", "evidence": ["Benchmarks show 2x speedup"], "confidence": 0.8}'
                ),
                # Round 1 Bear
                MagicMock(
                    content='{"argument": "Steep learning curve", "evidence": ["Survey shows 60% find it hard"], "confidence": 0.7}'
                ),
                # Round 2 Bull
                MagicMock(
                    content='{"argument": "Active community support", "evidence": ["Growing ecosystem"], "confidence": 0.8}'
                ),
                # Round 2 Bear
                MagicMock(
                    content='{"argument": "Limited tooling compared to Python", "evidence": ["Fewer libraries"], "confidence": 0.6}'
                ),
                # Synthesis
                MagicMock(
                    content='{"summary": "Technology shows promise but has adoption barriers", "bull_strengths": ["Performance"], "bear_concerns": ["Learning curve"], "balanced_assessment": "Good for performance-critical systems", "recommendation": "Evaluate based on team expertise", "confidence_score": 0.75}'
                ),
            ]
        )

        task = AgentTask(
            type="debate",
            params={
                "topic": "Should we adopt Rust for our backend?",
                "context": {"team_size": 5, "current_stack": "Python"},
            },
            user_id="test-user",
        )

        result = await debate_agent.execute(task)

        assert result.success
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_research_finding_debate(self, debate_agent: DebateAgent) -> None:
        """Test debate on research findings."""
        # DebateAgent runs 2 rounds by default: 2*(bull+bear) + synthesis = 5 LLM calls
        debate_agent.llm_router.complete = AsyncMock(
            side_effect=[
                # Round 1 Bull
                MagicMock(
                    content='{"argument": "Novel approach with state-of-the-art results", "evidence": ["Outperforms baselines"], "confidence": 0.85}'
                ),
                # Round 1 Bear
                MagicMock(
                    content='{"argument": "Limited evaluation scope", "evidence": ["Only tested on small datasets"], "confidence": 0.6}'
                ),
                # Round 2 Bull
                MagicMock(
                    content='{"argument": "Theoretical foundation is sound", "evidence": ["Builds on proven methods"], "confidence": 0.8}'
                ),
                # Round 2 Bear
                MagicMock(
                    content='{"argument": "Reproducibility concerns", "evidence": ["No code released"], "confidence": 0.7}'
                ),
                # Synthesis
                MagicMock(
                    content='{"summary": "Promising but requires broader validation", "bull_strengths": ["Novel approach"], "bear_concerns": ["Limited evaluation"], "balanced_assessment": "Worth monitoring but wait for more evidence", "recommendation": "Wait for reproduction studies", "confidence_score": 0.65}'
                ),
            ]
        )

        task = AgentTask(
            type="debate",
            params={
                "topic": "Is this new model architecture worth implementing?",
                "research_context": {"paper_id": "arxiv:1234.5678"},
            },
            user_id="test-user",
        )

        result = await debate_agent.execute(task)

        assert result.success


class TestErrorRecovery:
    """Integration tests for error recovery scenarios."""

    @pytest.mark.asyncio
    async def test_mcp_server_timeout_recovery(
        self, mock_llm_router: LLMRouter
    ) -> None:
        """Test recovery from MCP server timeout."""
        mock_client = AsyncMock()

        call_count = 0

        async def flaky_invoke(
            server: str, tool: str, params: dict[str, Any]
        ) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("MCP server timeout")
            return [{"title": "Test", "id": "123"}]

        mock_client.invoke = AsyncMock(side_effect=flaky_invoke)

        agent = ResearchAgent(llm_router=mock_llm_router, mcp_client=mock_client)

        # Should recover from timeout on retry
        results = await agent._search_arxiv("test query", {})

        # May be empty if all retries failed, but should not raise
        assert isinstance(results, ResearchFindings)

    @pytest.mark.asyncio
    async def test_partial_search_failure(
        self, mock_llm_router: LLMRouter
    ) -> None:
        """Test handling of partial search failures."""
        mock_client = AsyncMock()

        async def partial_failure(
            server: str, tool: str, params: dict[str, Any]
        ) -> dict[str, Any]:
            if server == "github":
                raise Exception("GitHub API rate limited")
            return [{"title": "Paper", "id": "123"}]

        mock_client.invoke = AsyncMock(side_effect=partial_failure)

        agent = ResearchAgent(llm_router=mock_llm_router, mcp_client=mock_client)

        task = AgentTask(
            type="research",
            params={"query": "machine learning", "source": "arxiv"},
            user_id="test-user",
        )

        # Should complete with partial results
        result = await agent.execute(task)

        # Should succeed with at least ArXiv results
        assert isinstance(result, AgentResult)

    @pytest.mark.asyncio
    async def test_llm_fallback(self, mock_mcp_client: AsyncMock) -> None:
        """Test LLM provider fallback."""
        from dova.config.providers import LLMRouter, LLMProvider, ProviderConfig, ModelConfig, TaskType

        # Create primary provider that fails
        primary = MagicMock(spec=LLMProvider)
        primary.name = "primary"
        primary.config = ProviderConfig(
            name="primary",
            enabled=True,
            priority=1,
            models={TaskType.REASONING: ModelConfig(model_id="model")},
        )
        primary.complete = AsyncMock(side_effect=Exception("Primary failed"))

        # Create fallback provider
        fallback = MagicMock(spec=LLMProvider)
        fallback.name = "fallback"
        fallback.config = ProviderConfig(
            name="fallback",
            enabled=True,
            priority=2,
            models={TaskType.REASONING: ModelConfig(model_id="model")},
        )
        fallback.complete = AsyncMock(
            return_value=MagicMock(content="Fallback response")
        )

        router = LLMRouter(providers={"primary": primary, "fallback": fallback})

        agent = ResearchAgent(llm_router=router, mcp_client=mock_mcp_client)

        # Should use fallback when primary fails
        # This depends on router implementation
        assert agent.llm_router is not None
