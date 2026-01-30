"""
Unit Tests for DOVA Tool Resolver.
"""

import pytest
from unittest.mock import MagicMock

from dova.services.tool_resolver import (
    TaskAnalyzer,
    TaskRequirements,
    ToolCategory,
    ToolPlan,
    ToolResolver,
    ToolSpec,
)


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create mock settings."""
    settings = MagicMock()
    settings.sandbox = MagicMock()
    settings.sandbox.enabled = True
    settings.aws = MagicMock()
    settings.aws.agentcore_memory_enabled = True
    return settings


@pytest.fixture
def mock_mcp_registry() -> MagicMock:
    """Create mock MCP registry."""
    registry = MagicMock()

    # Create mock server with tools
    mock_server = MagicMock()
    mock_server.name = "arxiv"
    mock_server.enabled = True
    mock_server.env_vars = {}

    mock_tool = MagicMock()
    mock_tool.name = "search"
    mock_tool.description = "Search ArXiv papers"
    mock_tool.capabilities = []
    mock_tool.input_schema = {}

    mock_server.tools = [mock_tool]
    registry.get_enabled_servers.return_value = [mock_server]

    return registry


@pytest.fixture
def mock_sandbox_executor() -> MagicMock:
    """Create mock sandbox executor."""
    executor = MagicMock()
    executor.is_available = MagicMock(return_value=True)
    return executor


@pytest.fixture
def mock_memory_service() -> MagicMock:
    """Create mock memory service."""
    return MagicMock()


@pytest.fixture
def task_analyzer() -> TaskAnalyzer:
    """Create task analyzer instance."""
    return TaskAnalyzer()


@pytest.fixture
def tool_resolver(
    mock_settings: MagicMock,
    mock_mcp_registry: MagicMock,
    mock_sandbox_executor: MagicMock,
    mock_memory_service: MagicMock,
) -> ToolResolver:
    """Create tool resolver instance."""
    return ToolResolver(
        settings=mock_settings,
        mcp_registry=mock_mcp_registry,
        sandbox_executor=mock_sandbox_executor,
        memory_service=mock_memory_service,
    )


class TestToolCategory:
    """Test cases for ToolCategory enum."""

    def test_all_categories_defined(self) -> None:
        """Test all tool categories are defined."""
        assert ToolCategory.SEARCH.value == "search"
        assert ToolCategory.EXECUTE.value == "execute"
        assert ToolCategory.VALIDATE.value == "validate"
        assert ToolCategory.SYNTHESIZE.value == "synthesize"
        assert ToolCategory.RECOMMEND.value == "recommend"
        assert ToolCategory.PROFILE.value == "profile"
        assert ToolCategory.MEMORY.value == "memory"
        assert ToolCategory.WEB.value == "web"


class TestToolSpec:
    """Test cases for ToolSpec dataclass."""

    def test_tool_spec_creation(self) -> None:
        """Test creating a tool specification."""
        spec = ToolSpec(
            name="mcp:arxiv:search",
            description="Search ArXiv papers",
            category=ToolCategory.SEARCH,
            source="mcp",
            capabilities=["paper_search", "academic"],
            cost_tier="free",
        )

        assert spec.name == "mcp:arxiv:search"
        assert spec.category == ToolCategory.SEARCH
        assert spec.source == "mcp"
        assert "paper_search" in spec.capabilities
        assert spec.cost_tier == "free"

    def test_tool_spec_defaults(self) -> None:
        """Test tool spec default values."""
        spec = ToolSpec(
            name="test_tool",
            description="Test",
            category=ToolCategory.SYNTHESIZE,
            source="internal",
        )

        assert spec.capabilities == []
        assert spec.enabled is True
        assert spec.requires_auth is False
        assert spec.cost_tier == "free"


class TestTaskRequirements:
    """Test cases for TaskRequirements dataclass."""

    def test_task_requirements_creation(self) -> None:
        """Test creating task requirements."""
        reqs = TaskRequirements(
            categories=[ToolCategory.SEARCH, ToolCategory.EXECUTE],
            keywords=["transformer", "attention"],
            requires_search=True,
            requires_execution=True,
            complexity="moderate",
        )

        assert ToolCategory.SEARCH in reqs.categories
        assert "transformer" in reqs.keywords
        assert reqs.requires_search is True
        assert reqs.complexity == "moderate"

    def test_task_requirements_defaults(self) -> None:
        """Test task requirements default values."""
        reqs = TaskRequirements(
            categories=[],
            keywords=[],
        )

        assert reqs.requires_search is False
        assert reqs.requires_execution is False
        assert reqs.complexity == "simple"
        assert reqs.estimated_tools == 1


class TestToolPlan:
    """Test cases for ToolPlan dataclass."""

    def test_tool_plan_creation(self) -> None:
        """Test creating a tool plan."""
        tool1 = ToolSpec(
            name="search_tool",
            description="Search",
            category=ToolCategory.SEARCH,
            source="mcp",
        )
        tool2 = ToolSpec(
            name="execute_tool",
            description="Execute",
            category=ToolCategory.EXECUTE,
            source="sandbox",
        )

        reqs = TaskRequirements(
            categories=[ToolCategory.SEARCH, ToolCategory.EXECUTE],
            keywords=["test"],
            requires_search=True,
            requires_execution=True,
        )

        plan = ToolPlan(
            task="Find and run code",
            requirements=reqs,
            selected_tools=[tool1, tool2],
            execution_order=["search_tool", "execute_tool"],
            fallback_tools=[],
        )

        assert len(plan.selected_tools) == 2
        assert plan.execution_order[0] == "search_tool"
        assert plan.task == "Find and run code"


class TestTaskAnalyzer:
    """Test cases for TaskAnalyzer."""

    def test_analyze_search_task(self, task_analyzer: TaskAnalyzer) -> None:
        """Test analyzing a search task."""
        reqs = task_analyzer.analyze(
            "Find papers about transformer attention mechanisms"
        )

        assert isinstance(reqs, TaskRequirements)
        assert reqs.requires_search is True
        assert ToolCategory.SEARCH in reqs.categories

    def test_analyze_code_task(self, task_analyzer: TaskAnalyzer) -> None:
        """Test analyzing a code execution task."""
        reqs = task_analyzer.analyze(
            "Run this Python code: print('hello')"
        )

        assert reqs.requires_execution is True
        assert ToolCategory.EXECUTE in reqs.categories

    def test_analyze_memory_task(self, task_analyzer: TaskAnalyzer) -> None:
        """Test analyzing a memory-related task."""
        reqs = task_analyzer.analyze(
            "Remember that I prefer PyTorch over TensorFlow"
        )

        assert reqs.requires_memory is True
        assert ToolCategory.MEMORY in reqs.categories

    def test_analyze_web_task(self, task_analyzer: TaskAnalyzer) -> None:
        """Test analyzing a web access task."""
        reqs = task_analyzer.analyze(
            "Fetch the documentation from the website URL"
        )

        assert reqs.requires_web is True
        assert ToolCategory.WEB in reqs.categories

    def test_analyze_validation_task(self, task_analyzer: TaskAnalyzer) -> None:
        """Test analyzing a validation task."""
        reqs = task_analyzer.analyze(
            "Validate this implementation and check for bugs"
        )

        assert ToolCategory.VALIDATE in reqs.categories

    def test_analyze_complex_task(self, task_analyzer: TaskAnalyzer) -> None:
        """Test analyzing a complex multi-capability task."""
        reqs = task_analyzer.analyze(
            "Search for transformer papers, run the example code, "
            "and remember my preferences for future recommendations"
        )

        # Should detect multiple categories
        assert len(reqs.categories) >= 2

    def test_keyword_extraction(self, task_analyzer: TaskAnalyzer) -> None:
        """Test keyword extraction from task."""
        reqs = task_analyzer.analyze(
            "Find papers about neural networks and transformers"
        )

        assert len(reqs.keywords) > 0
        # Should extract meaningful words
        assert "neural" in reqs.keywords or "networks" in reqs.keywords

    def test_complexity_simple_task(self, task_analyzer: TaskAnalyzer) -> None:
        """Test complexity score for simple task."""
        reqs = task_analyzer.analyze("Find papers")

        assert reqs.complexity == "simple"

    def test_complexity_complex_task(self, task_analyzer: TaskAnalyzer) -> None:
        """Test complexity score for complex task."""
        reqs = task_analyzer.analyze(
            "Search for papers, execute the code examples, validate results, "
            "compare with existing implementations, summarize findings, "
            "and fetch additional documentation from the web"
        )

        assert reqs.complexity in ["moderate", "complex"]

    def test_reasoning_generated(self, task_analyzer: TaskAnalyzer) -> None:
        """Test that reasoning is generated."""
        reqs = task_analyzer.analyze("Search for machine learning papers")

        assert len(reqs.reasoning) > 0
        assert "search" in reqs.reasoning.lower() or "sources" in reqs.reasoning.lower()


class TestToolResolver:
    """Test cases for ToolResolver."""

    def test_initialization(self, tool_resolver: ToolResolver) -> None:
        """Test tool resolver initialization."""
        assert tool_resolver.settings is not None
        assert tool_resolver.mcp_registry is not None

    def test_discover_tools(self, tool_resolver: ToolResolver) -> None:
        """Test tool discovery."""
        tools = tool_resolver.discover_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_discover_mcp_tools(self, tool_resolver: ToolResolver) -> None:
        """Test MCP tool discovery."""
        tools = tool_resolver._discover_mcp_tools()

        assert isinstance(tools, list)
        # Should include arxiv from mock registry
        tool_names = [t.name for t in tools]
        assert any("arxiv" in name for name in tool_names)

    def test_discover_sandbox_tools(self, tool_resolver: ToolResolver) -> None:
        """Test sandbox tool discovery."""
        tools = tool_resolver._discover_sandbox_tools()

        assert isinstance(tools, list)
        # Should have sandbox tools when enabled
        assert len(tools) > 0
        assert all(t.source == "sandbox" for t in tools)

    def test_discover_recommendation_tools(self, tool_resolver: ToolResolver) -> None:
        """Test recommendation tool discovery."""
        tools = tool_resolver._discover_recommendation_tools()

        assert isinstance(tools, list)
        assert len(tools) > 0
        assert all(t.category == ToolCategory.RECOMMEND for t in tools)

    def test_discover_internal_tools(self, tool_resolver: ToolResolver) -> None:
        """Test internal tool discovery."""
        tools = tool_resolver._discover_internal_tools()

        assert isinstance(tools, list)
        assert all(t.source == "internal" for t in tools)

    def test_resolve_tools_for_search(self, tool_resolver: ToolResolver) -> None:
        """Test resolving tools for search requirements."""
        tool_resolver.discover_tools()

        reqs = TaskRequirements(
            categories=[ToolCategory.SEARCH],
            keywords=["papers", "arxiv"],
            requires_search=True,
        )

        tools = tool_resolver.resolve_tools(reqs)

        assert isinstance(tools, list)
        # Should prioritize search tools
        if tools:
            assert tools[0].category == ToolCategory.SEARCH

    def test_resolve_tools_max_limit(self, tool_resolver: ToolResolver) -> None:
        """Test resolving tools respects max_tools limit."""
        tool_resolver.discover_tools()

        reqs = TaskRequirements(
            categories=[ToolCategory.SEARCH, ToolCategory.EXECUTE],
            keywords=["test"],
            requires_search=True,
        )

        tools = tool_resolver.resolve_tools(reqs, max_tools=2)

        assert len(tools) <= 2

    def test_create_plan(self, tool_resolver: ToolResolver) -> None:
        """Test creating a tool execution plan."""
        plan = tool_resolver.create_plan(
            task="Find papers about attention mechanisms",
            context=None,
        )

        assert isinstance(plan, ToolPlan)
        assert plan.requirements is not None
        assert isinstance(plan.selected_tools, list)
        assert isinstance(plan.execution_order, list)
        assert plan.task == "Find papers about attention mechanisms"

    def test_create_plan_with_context(self, tool_resolver: ToolResolver) -> None:
        """Test creating plan with context."""
        plan = tool_resolver.create_plan(
            task="Search for models",
            context={"user_preference": "huggingface"},
        )

        assert isinstance(plan, ToolPlan)
        assert plan.requirements is not None

    def test_execution_order(self, tool_resolver: ToolResolver) -> None:
        """Test execution order prioritizes correctly."""
        tool_resolver.discover_tools()

        tools = [
            ToolSpec(name="exec", description="", category=ToolCategory.EXECUTE, source="test"),
            ToolSpec(name="search", description="", category=ToolCategory.SEARCH, source="test"),
            ToolSpec(name="synth", description="", category=ToolCategory.SYNTHESIZE, source="test"),
        ]

        reqs = TaskRequirements(categories=[], keywords=[])
        order = tool_resolver._determine_execution_order(tools, reqs)

        # Search should come before execute, which should come before synthesize
        search_idx = order.index("search")
        exec_idx = order.index("exec")
        synth_idx = order.index("synth")

        assert search_idx < exec_idx < synth_idx

    def test_score_tool_category_match(self, tool_resolver: ToolResolver) -> None:
        """Test tool scoring for category match."""
        tool = ToolSpec(
            name="mcp:arxiv:search",
            description="Search papers",
            category=ToolCategory.SEARCH,
            source="mcp",
            capabilities=["search", "academic"],
        )

        reqs = TaskRequirements(
            categories=[ToolCategory.SEARCH],
            keywords=["papers"],
            requires_search=True,
        )

        score = tool_resolver._score_tool(tool, reqs)

        assert score > 0  # Should have positive score for matching category

    def test_score_tool_no_match(self, tool_resolver: ToolResolver) -> None:
        """Test scoring tool with no category match."""
        tool = ToolSpec(
            name="execute_tool",
            description="Execute code",
            category=ToolCategory.EXECUTE,
            source="sandbox",
            capabilities=["execute", "python"],
        )

        reqs = TaskRequirements(
            categories=[ToolCategory.SEARCH],
            keywords=["papers"],
            requires_search=True,
        )

        score = tool_resolver._score_tool(tool, reqs)

        # Should still have some score from cost tier bonus
        assert score >= 0

    def test_get_tool(self, tool_resolver: ToolResolver) -> None:
        """Test getting a specific tool by name."""
        tool_resolver.discover_tools()

        # Should find sandbox:execute tool
        tool = tool_resolver.get_tool("sandbox:execute")

        assert tool is not None
        assert tool.name == "sandbox:execute"

    def test_get_tool_not_found(self, tool_resolver: ToolResolver) -> None:
        """Test getting a non-existent tool."""
        tool_resolver.discover_tools()

        tool = tool_resolver.get_tool("nonexistent:tool")

        assert tool is None

    def test_get_tools_by_category(self, tool_resolver: ToolResolver) -> None:
        """Test getting tools by category."""
        tool_resolver.discover_tools()

        search_tools = tool_resolver.get_tools_by_category(ToolCategory.SEARCH)

        assert isinstance(search_tools, list)
        assert all(t.category == ToolCategory.SEARCH for t in search_tools)


class TestToolResolverEdgeCases:
    """Test edge cases for ToolResolver."""

    def test_resolver_sandbox_disabled(self, mock_mcp_registry: MagicMock) -> None:
        """Test resolver when sandbox is disabled."""
        settings = MagicMock()
        settings.sandbox = MagicMock()
        settings.sandbox.enabled = False
        settings.aws = MagicMock()
        settings.aws.agentcore_memory_enabled = False

        resolver = ToolResolver(
            settings=settings,
            mcp_registry=mock_mcp_registry,
        )
        tools = resolver.discover_tools()

        # Should not include sandbox tools
        sandbox_tools = [t for t in tools if t.source == "sandbox"]
        assert len(sandbox_tools) == 0

    def test_resolve_empty_requirements(self, tool_resolver: ToolResolver) -> None:
        """Test resolving with empty requirements."""
        tool_resolver.discover_tools()

        reqs = TaskRequirements(
            categories=[],
            keywords=[],
        )

        tools = tool_resolver.resolve_tools(reqs)

        # Should still return tools (sorted by cost tier)
        assert isinstance(tools, list)

    def test_create_plan_empty_task(self, tool_resolver: ToolResolver) -> None:
        """Test creating plan for empty task."""
        plan = tool_resolver.create_plan(task="", context=None)

        assert isinstance(plan, ToolPlan)
        # Should handle gracefully
        assert plan.requirements is not None

    def test_fallback_tools_identified(self, tool_resolver: ToolResolver) -> None:
        """Test that fallback tools are identified."""
        plan = tool_resolver.create_plan(
            task="Search for papers and execute code",
            context=None,
        )

        # Fallback tools list should exist
        assert isinstance(plan.fallback_tools, list)

    def test_plan_reasoning_generated(self, tool_resolver: ToolResolver) -> None:
        """Test that plan reasoning is generated."""
        plan = tool_resolver.create_plan(
            task="Search for machine learning papers",
            context=None,
        )

        assert len(plan.reasoning) > 0
