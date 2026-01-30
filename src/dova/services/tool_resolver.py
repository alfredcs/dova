"""
Tool Resolver for DOVA.

Proactively discovers, resolves, and manages tools based on task requirements
and environment configuration.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

import structlog

from dova.config.mcp_servers import MCPCapability, MCPRegistry, MCPTool, get_default_registry
from dova.config.settings import Settings, get_settings

logger = structlog.get_logger(__name__)


class ToolCategory(Enum):
    """Categories of tools available in DOVA."""

    SEARCH = "search"  # Search external sources (ArXiv, GitHub, HuggingFace)
    EXECUTE = "execute"  # Code execution in sandbox
    VALIDATE = "validate"  # Code validation and analysis
    SYNTHESIZE = "synthesize"  # Result synthesis and summarization
    RECOMMEND = "recommend"  # Proactive recommendations
    PROFILE = "profile"  # User profiling and preferences
    MEMORY = "memory"  # Memory and knowledge storage
    WEB = "web"  # Web scraping and API access


@dataclass
class ToolSpec:
    """Specification of an available tool."""

    name: str
    category: ToolCategory
    description: str
    source: str  # "mcp", "sandbox", "internal", "custom"
    enabled: bool = True
    requires_auth: bool = False
    cost_tier: str = "free"  # "free", "low", "medium", "high"
    capabilities: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    handler: Callable[..., Coroutine[Any, Any, Any]] | None = None


@dataclass
class TaskRequirements:
    """Analyzed requirements for a task."""

    categories: list[ToolCategory]
    keywords: list[str]
    requires_search: bool = False
    requires_execution: bool = False
    requires_web: bool = False
    requires_memory: bool = False
    complexity: str = "simple"  # "simple", "moderate", "complex"
    estimated_tools: int = 1
    reasoning: str = ""


@dataclass
class ToolPlan:
    """Plan for which tools to use for a task."""

    task: str
    requirements: TaskRequirements
    selected_tools: list[ToolSpec]
    execution_order: list[str]
    fallback_tools: list[ToolSpec]
    reasoning: str = ""


class TaskAnalyzer:
    """
    Analyzes tasks to determine required capabilities and tools.

    Uses pattern matching and heuristics to understand task requirements.
    """

    # Patterns indicating different capabilities needed
    SEARCH_PATTERNS = [
        r"\b(search|find|look\s+for|discover|explore)\b",
        r"\b(papers?|articles?|research|publications?)\b",
        r"\b(repositories?|repos?|projects?|code)\b",
        r"\b(models?|datasets?|huggingface|arxiv|github)\b",
        r"\b(latest|recent|new|trending)\b",
    ]

    EXECUTION_PATTERNS = [
        r"\b(run|execute|test|evaluate|benchmark)\b",
        r"\b(code|script|program|function)\b",
        r"\bpython|javascript|node|go|rust\b",
        r"\b(output|result|performance)\b",
    ]

    VALIDATION_PATTERNS = [
        r"\b(validate|verify|check|analyze|review)\b",
        r"\b(quality|security|bugs?|issues?|vulnerabilities?)\b",
        r"\b(best\s+practices?|standards?|lint)\b",
    ]

    SYNTHESIS_PATTERNS = [
        r"\b(summarize|synthesize|combine|aggregate)\b",
        r"\b(compare|contrast|evaluate|assess)\b",
        r"\b(insights?|conclusions?|recommendations?)\b",
    ]

    WEB_PATTERNS = [
        r"\b(web|website|url|http|api)\b",
        r"\b(scrape|fetch|download|retrieve)\b",
        r"\b(documentation|docs|reference)\b",
    ]

    MEMORY_PATTERNS = [
        r"\b(remember|recall|history|previous|past)\b",
        r"\b(save|store|knowledge|learned)\b",
        r"\b(profile|preferences?|interests?)\b",
    ]

    def __init__(self) -> None:
        self._logger = logger.bind(service="task_analyzer")
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        self._patterns = {
            ToolCategory.SEARCH: [re.compile(p, re.I) for p in self.SEARCH_PATTERNS],
            ToolCategory.EXECUTE: [re.compile(p, re.I) for p in self.EXECUTION_PATTERNS],
            ToolCategory.VALIDATE: [re.compile(p, re.I) for p in self.VALIDATION_PATTERNS],
            ToolCategory.SYNTHESIZE: [re.compile(p, re.I) for p in self.SYNTHESIS_PATTERNS],
            ToolCategory.WEB: [re.compile(p, re.I) for p in self.WEB_PATTERNS],
            ToolCategory.MEMORY: [re.compile(p, re.I) for p in self.MEMORY_PATTERNS],
        }

    def analyze(self, task: str, context: dict[str, Any] | None = None) -> TaskRequirements:
        """
        Analyze a task to determine requirements.

        Args:
            task: Task description
            context: Optional context with hints

        Returns:
            TaskRequirements with identified needs
        """
        categories: list[ToolCategory] = []
        keywords: list[str] = []
        scores: dict[ToolCategory, int] = {}

        # Score each category based on pattern matches
        for category, patterns in self._patterns.items():
            score = sum(1 for p in patterns if p.search(task))
            if score > 0:
                scores[category] = score
                categories.append(category)

        # Extract keywords (simple noun extraction)
        words = re.findall(r"\b[a-zA-Z]{3,}\b", task.lower())
        stop_words = {"the", "and", "for", "with", "that", "this", "from", "are", "was", "have"}
        keywords = [w for w in words if w not in stop_words][:10]

        # Determine complexity
        if len(categories) >= 3 or len(task) > 500:
            complexity = "complex"
        elif len(categories) >= 2 or len(task) > 200:
            complexity = "moderate"
        else:
            complexity = "simple"

        # Build requirements
        requirements = TaskRequirements(
            categories=sorted(categories, key=lambda c: scores.get(c, 0), reverse=True),
            keywords=keywords,
            requires_search=ToolCategory.SEARCH in categories,
            requires_execution=ToolCategory.EXECUTE in categories,
            requires_web=ToolCategory.WEB in categories,
            requires_memory=ToolCategory.MEMORY in categories,
            complexity=complexity,
            estimated_tools=max(1, len(categories)),
            reasoning=self._generate_reasoning(task, categories, scores),
        )

        self._logger.debug(
            "task_analyzed",
            categories=[c.value for c in categories],
            complexity=complexity,
            keywords=keywords[:5],
        )

        return requirements

    def _generate_reasoning(
        self,
        task: str,
        categories: list[ToolCategory],
        scores: dict[ToolCategory, int],
    ) -> str:
        """Generate reasoning for the analysis."""
        parts = []
        if ToolCategory.SEARCH in categories:
            parts.append("Task requires searching external sources")
        if ToolCategory.EXECUTE in categories:
            parts.append("Task involves code execution")
        if ToolCategory.VALIDATE in categories:
            parts.append("Task requires validation or analysis")
        if ToolCategory.SYNTHESIZE in categories:
            parts.append("Task needs synthesis of information")
        if not parts:
            parts.append("Task appears to be a general reasoning task")
        return ". ".join(parts) + "."


class ToolResolver:
    """
    Resolves and provides access to tools based on configuration and requirements.

    Discovers available tools from:
    - MCP servers (ArXiv, GitHub, HuggingFace)
    - Sandbox execution service
    - Recommendation services
    - Internal DOVA services
    """

    def __init__(
        self,
        settings: Settings | None = None,
        mcp_registry: MCPRegistry | None = None,
        sandbox_executor: Any | None = None,
        memory_service: Any | None = None,
    ):
        self.settings = settings or get_settings()
        self.mcp_registry = mcp_registry or get_default_registry()
        self.sandbox_executor = sandbox_executor
        self.memory_service = memory_service
        self._tool_cache: dict[str, ToolSpec] = {}
        self._analyzer = TaskAnalyzer()
        self._logger = logger.bind(service="tool_resolver")

    def discover_tools(self) -> list[ToolSpec]:
        """
        Discover all available tools based on current configuration.

        Returns:
            List of available ToolSpecs
        """
        tools: list[ToolSpec] = []

        # Discover MCP tools
        tools.extend(self._discover_mcp_tools())

        # Discover sandbox tools
        tools.extend(self._discover_sandbox_tools())

        # Discover recommendation tools
        tools.extend(self._discover_recommendation_tools())

        # Discover internal tools
        tools.extend(self._discover_internal_tools())

        # Cache tools
        for tool in tools:
            self._tool_cache[tool.name] = tool

        self._logger.info("tools_discovered", count=len(tools))
        return tools

    def _discover_mcp_tools(self) -> list[ToolSpec]:
        """Discover tools from MCP servers."""
        tools: list[ToolSpec] = []

        for server in self.mcp_registry.get_enabled_servers():
            for mcp_tool in server.tools:
                category = self._mcp_capability_to_category(mcp_tool.capabilities)
                tools.append(
                    ToolSpec(
                        name=f"mcp:{server.name}:{mcp_tool.name}",
                        category=category,
                        description=mcp_tool.description,
                        source="mcp",
                        enabled=server.enabled,
                        requires_auth=bool(server.env_vars),
                        capabilities=[c.value for c in mcp_tool.capabilities],
                        parameters=mcp_tool.input_schema,
                    )
                )

        return tools

    def _discover_sandbox_tools(self) -> list[ToolSpec]:
        """Discover sandbox execution tools."""
        tools: list[ToolSpec] = []

        if self.settings.sandbox.enabled:
            tools.append(
                ToolSpec(
                    name="sandbox:execute",
                    category=ToolCategory.EXECUTE,
                    description="Execute code in isolated Docker container",
                    source="sandbox",
                    enabled=True,
                    cost_tier="low",
                    capabilities=["execute", "validate"],
                    parameters={
                        "code": {"type": "string", "required": True},
                        "language": {"type": "string", "default": "python"},
                        "dependencies": {"type": "array", "items": {"type": "string"}},
                        "tier": {"type": "string", "enum": ["cpu_basic", "cpu_standard", "gpu_spot"]},
                    },
                )
            )

            tools.append(
                ToolSpec(
                    name="sandbox:validate",
                    category=ToolCategory.VALIDATE,
                    description="Validate code with static analysis and optional execution",
                    source="sandbox",
                    enabled=True,
                    cost_tier="free",
                    capabilities=["validate", "analyze"],
                )
            )

        return tools

    def _discover_recommendation_tools(self) -> list[ToolSpec]:
        """Discover proactive recommendation tools."""
        tools: list[ToolSpec] = []

        # These are always available (even if services aren't initialized)
        tools.append(
            ToolSpec(
                name="recommend:subscribe",
                category=ToolCategory.RECOMMEND,
                description="Subscribe to content categories for proactive recommendations",
                source="internal",
                enabled=True,
                capabilities=["subscribe", "personalize"],
            )
        )

        tools.append(
            ToolSpec(
                name="recommend:get",
                category=ToolCategory.RECOMMEND,
                description="Get personalized content recommendations",
                source="internal",
                enabled=True,
                capabilities=["recommend", "personalize"],
            )
        )

        return tools

    def _discover_internal_tools(self) -> list[ToolSpec]:
        """Discover internal DOVA tools."""
        tools: list[ToolSpec] = []

        # Memory tools
        if self.memory_service or self.settings.aws.agentcore_memory_enabled:
            tools.append(
                ToolSpec(
                    name="memory:store",
                    category=ToolCategory.MEMORY,
                    description="Store information in long-term memory",
                    source="internal",
                    enabled=True,
                    capabilities=["store", "persist"],
                )
            )
            tools.append(
                ToolSpec(
                    name="memory:recall",
                    category=ToolCategory.MEMORY,
                    description="Recall information from memory",
                    source="internal",
                    enabled=True,
                    capabilities=["recall", "search"],
                )
            )

        # Profile tools
        tools.append(
            ToolSpec(
                name="profile:get",
                category=ToolCategory.PROFILE,
                description="Get user profile and preferences",
                source="internal",
                enabled=True,
                capabilities=["profile", "preferences"],
            )
        )

        # Synthesis tools (always available)
        tools.append(
            ToolSpec(
                name="synthesize:combine",
                category=ToolCategory.SYNTHESIZE,
                description="Synthesize multiple results into coherent output",
                source="internal",
                enabled=True,
                capabilities=["synthesize", "aggregate"],
            )
        )

        return tools

    def _mcp_capability_to_category(self, capabilities: list[MCPCapability]) -> ToolCategory:
        """Map MCP capabilities to tool category."""
        if MCPCapability.SEARCH in capabilities:
            return ToolCategory.SEARCH
        elif MCPCapability.FETCH in capabilities:
            return ToolCategory.WEB
        elif MCPCapability.CREATE in capabilities or MCPCapability.UPDATE in capabilities:
            return ToolCategory.MEMORY
        return ToolCategory.SEARCH

    def resolve_tools(
        self,
        requirements: TaskRequirements,
        max_tools: int = 5,
    ) -> list[ToolSpec]:
        """
        Resolve best tools for given requirements.

        Args:
            requirements: Analyzed task requirements
            max_tools: Maximum number of tools to return

        Returns:
            List of recommended tools in priority order
        """
        if not self._tool_cache:
            self.discover_tools()

        # Score each tool based on requirements match
        scored_tools: list[tuple[ToolSpec, float]] = []

        for tool in self._tool_cache.values():
            if not tool.enabled:
                continue

            score = self._score_tool(tool, requirements)
            if score > 0:
                scored_tools.append((tool, score))

        # Sort by score descending
        scored_tools.sort(key=lambda x: x[1], reverse=True)

        # Return top tools
        selected = [t for t, _ in scored_tools[:max_tools]]

        self._logger.debug(
            "tools_resolved",
            requirements_categories=[c.value for c in requirements.categories],
            selected_tools=[t.name for t in selected],
        )

        return selected

    def _score_tool(self, tool: ToolSpec, requirements: TaskRequirements) -> float:
        """Score a tool based on how well it matches requirements."""
        score = 0.0

        # Category match (highest weight)
        if tool.category in requirements.categories:
            category_rank = requirements.categories.index(tool.category)
            score += (len(requirements.categories) - category_rank) * 2.0

        # Specific requirement matches
        if requirements.requires_search and tool.category == ToolCategory.SEARCH:
            score += 1.5
        if requirements.requires_execution and tool.category == ToolCategory.EXECUTE:
            score += 1.5
        if requirements.requires_web and tool.category == ToolCategory.WEB:
            score += 1.0
        if requirements.requires_memory and tool.category == ToolCategory.MEMORY:
            score += 1.0

        # Keyword matches in capabilities
        for keyword in requirements.keywords:
            for cap in tool.capabilities:
                if keyword.lower() in cap.lower():
                    score += 0.5

        # Cost tier preference (prefer cheaper tools)
        cost_bonus = {"free": 0.3, "low": 0.2, "medium": 0.1, "high": 0.0}
        score += cost_bonus.get(tool.cost_tier, 0.0)

        return score

    def create_plan(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> ToolPlan:
        """
        Create a complete tool execution plan for a task.

        Args:
            task: Task description
            context: Optional context

        Returns:
            ToolPlan with selected tools and execution order
        """
        # Analyze task
        requirements = self._analyzer.analyze(task, context)

        # Resolve tools
        tools = self.resolve_tools(requirements)

        # Determine execution order
        execution_order = self._determine_execution_order(tools, requirements)

        # Identify fallback tools
        fallback_tools = self._identify_fallbacks(tools, requirements)

        plan = ToolPlan(
            task=task,
            requirements=requirements,
            selected_tools=tools,
            execution_order=execution_order,
            fallback_tools=fallback_tools,
            reasoning=self._generate_plan_reasoning(requirements, tools),
        )

        self._logger.info(
            "tool_plan_created",
            task=task[:100],
            tools=[t.name for t in tools],
            execution_order=execution_order,
        )

        return plan

    def _determine_execution_order(
        self,
        tools: list[ToolSpec],
        requirements: TaskRequirements,
    ) -> list[str]:
        """Determine optimal execution order for tools."""
        # Priority order: search → validate → execute → synthesize → recommend
        priority = {
            ToolCategory.SEARCH: 1,
            ToolCategory.MEMORY: 2,
            ToolCategory.WEB: 3,
            ToolCategory.VALIDATE: 4,
            ToolCategory.EXECUTE: 5,
            ToolCategory.SYNTHESIZE: 6,
            ToolCategory.RECOMMEND: 7,
            ToolCategory.PROFILE: 8,
        }

        sorted_tools = sorted(tools, key=lambda t: priority.get(t.category, 10))
        return [t.name for t in sorted_tools]

    def _identify_fallbacks(
        self,
        selected_tools: list[ToolSpec],
        requirements: TaskRequirements,
    ) -> list[ToolSpec]:
        """Identify fallback tools if primary tools fail."""
        fallbacks: list[ToolSpec] = []
        selected_names = {t.name for t in selected_tools}

        for tool in self._tool_cache.values():
            if tool.name in selected_names:
                continue
            if not tool.enabled:
                continue

            # Add fallbacks for each category in requirements
            for category in requirements.categories:
                if tool.category == category and len(fallbacks) < 3:
                    fallbacks.append(tool)
                    break

        return fallbacks

    def _generate_plan_reasoning(
        self,
        requirements: TaskRequirements,
        tools: list[ToolSpec],
    ) -> str:
        """Generate reasoning for the tool plan."""
        parts = [f"Task complexity: {requirements.complexity}"]

        if tools:
            tool_desc = ", ".join(t.name.split(":")[-1] for t in tools[:3])
            parts.append(f"Primary tools: {tool_desc}")

        if requirements.requires_search:
            parts.append("Will search external sources for information")
        if requirements.requires_execution:
            parts.append("Code execution may be needed")

        return ". ".join(parts) + "."

    def get_tool(self, name: str) -> ToolSpec | None:
        """Get a specific tool by name."""
        if not self._tool_cache:
            self.discover_tools()
        return self._tool_cache.get(name)

    def get_tools_by_category(self, category: ToolCategory) -> list[ToolSpec]:
        """Get all tools in a category."""
        if not self._tool_cache:
            self.discover_tools()
        return [t for t in self._tool_cache.values() if t.category == category and t.enabled]
