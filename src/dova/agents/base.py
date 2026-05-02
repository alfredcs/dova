"""
Base Agent Implementation for DOVA.

Provides the foundation for all DOVA agents with unified interfaces
for LLM calls, MCP tool invocation, and lifecycle management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import structlog

from dova.agents.mixins.memory import MemoryMixin
from dova.agents.mixins.reasoning import ReasoningMixin
from dova.config.providers import LLMRequest, LLMResponse, LLMRouter, TaskType
from dova.utils.metrics import MetricsCollector, MetricNames
from dova.utils.retry import RetryConfig, retry_async

logger = structlog.get_logger(__name__)


class AgentStatus(Enum):
    """Status of an agent execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentTask:
    """A task to be executed by an agent."""

    id: str = field(default_factory=lambda: str(uuid4()))
    type: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    user_id: str | None = None
    priority: int = 5
    created_at: datetime = field(default_factory=datetime.utcnow)
    timeout_seconds: int = 300


@dataclass
class AgentResult:
    """Result of an agent execution."""

    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    agent_name: str = ""
    task_id: str = ""


@dataclass
class MCPToolResult:
    """Result from an MCP tool invocation."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    server: str = ""
    tool: str = ""
    cached: bool = False


class BaseAgent(ReasoningMixin, MemoryMixin, ABC):
    """
    Abstract base class for all DOVA agents.

    Provides common functionality for:
    - LLM interactions with retry and fallback
    - MCP tool invocations
    - Metrics collection
    - Structured logging
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        mcp_client: Any | None = None,
        metrics: MetricsCollector | None = None,
        retry_config: RetryConfig | None = None,
        memory_service: Any | None = None,
        session_manager: Any | None = None,
    ):
        self.llm_router = llm_router
        self.mcp_client = mcp_client
        self.metrics = metrics or MetricsCollector()
        self.retry_config = retry_config or RetryConfig(max_retries=3)
        self.memory_service = memory_service
        self.session_manager = session_manager  # AgentCore memory session manager
        self.name = self.__class__.__name__
        self._logger = logger.bind(agent=self.name)
        # Per-stage token accounting: last_*_tokens is refreshed every
        # think() call so orchestrators can compute stage-level totals.
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0

    @abstractmethod
    async def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute the agent's primary function.

        Args:
            task: The task to execute

        Returns:
            AgentResult containing the execution result
        """
        pass

    @property
    def system_prompt(self) -> str:
        """System prompt for this agent. Override in subclasses."""
        return f"You are {self.name}, an AI agent in the DOVA research platform."

    async def think(
        self,
        prompt: str,
        task_type: TaskType = TaskType.REASONING,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Make an LLM call for reasoning.

        Args:
            prompt: User message/prompt
            task_type: Type of task for model selection
            system_prompt: Override default system prompt
            temperature: Override temperature
            max_tokens: Override max tokens

        Returns:
            LLM response content
        """
        request = LLMRequest(
            task_type=task_type,
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt or self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        async def _call() -> LLMResponse:
            return await self.llm_router.complete(request)

        with self.metrics.timer(MetricNames.LLM_CALL_LATENCY, {"agent": self.name}):
            response = await retry_async(_call, config=self.retry_config)

        self.metrics.record(
            MetricNames.LLM_INPUT_TOKENS,
            response.input_tokens,
            labels={"agent": self.name},
        )
        self.metrics.record(
            MetricNames.LLM_OUTPUT_TOKENS,
            response.output_tokens,
            labels={"agent": self.name},
        )

        # Expose last-call token counts so orchestrators can do per-stage
        # accounting without changing the think() return shape.
        self.last_input_tokens = response.input_tokens
        self.last_output_tokens = response.output_tokens

        self._logger.debug(
            "llm_call_complete",
            provider=response.provider,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        return response.content

    async def think_stream(
        self,
        prompt: str,
        task_type: TaskType = TaskType.REASONING,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream an LLM call, yielding tokens as they arrive.

        Same interface as think() but returns an async iterator of token strings.
        """
        request = LLMRequest(
            task_type=task_type,
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt or self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for token in self.llm_router.stream(request):
            yield token

    async def call_tool(
        self,
        server: str,
        tool: str,
        params: dict[str, Any],
        cache_ttl: float = 600,
    ) -> MCPToolResult:
        """
        Invoke an MCP tool.

        Args:
            server: MCP server name (e.g., "arxiv", "github")
            tool: Tool name to invoke
            params: Tool parameters
            cache_ttl: Cache time-to-live in seconds (default 10 min)

        Returns:
            MCPToolResult with the tool response
        """
        if self.mcp_client is None:
            return MCPToolResult(
                success=False,
                error="MCP client not configured",
                server=server,
                tool=tool,
            )

        self._logger.debug("mcp_tool_call", server=server, tool=tool, params=params)

        try:
            with self.metrics.timer(
                MetricNames.MCP_CALL_LATENCY,
                {"server": server, "tool": tool},
            ):
                result = await self.mcp_client.invoke(server, tool, params, cache_ttl=cache_ttl)

            self.metrics.increment(
                MetricNames.MCP_CALL_COUNT,
                labels={"server": server, "tool": tool, "status": "success"},
            )

            return MCPToolResult(
                success=True,
                data=result,
                server=server,
                tool=tool,
            )

        except Exception as e:
            self._logger.warning(
                "mcp_tool_error",
                server=server,
                tool=tool,
                error=str(e),
            )
            self.metrics.increment(
                MetricNames.MCP_ERROR_COUNT,
                labels={"server": server, "tool": tool},
            )
            return MCPToolResult(
                success=False,
                error=str(e),
                server=server,
                tool=tool,
            )

    async def search_arxiv(
        self,
        query: str,
        max_results: int = 10,
        date_from: str | None = None,
    ) -> MCPToolResult:
        """Convenience method for ArXiv search.

        Args:
            query: Search query string.
            max_results: Max papers to return.
            date_from: Optional inclusive start date in ``YYYY-MM-DD`` format.
                Recent-only windowing keeps synthesis grounded in current work.
        """
        params: dict[str, Any] = {"query": query, "max_results": max_results}
        if date_from:
            params["date_from"] = date_from
        return await self.call_tool(
            "arxiv",
            "search_papers",  # blazickjp/arxiv-mcp-server uses this tool name
            params,
        )

    async def search_github(
        self,
        query: str,
        search_type: str = "repositories",
        per_page: int = 30,
        sort: str | None = None,
    ) -> MCPToolResult:
        """Convenience method for GitHub search.

        Args:
            query: Search query
            search_type: Type of search (repositories, code, etc.)
            per_page: Number of results per page
            sort: Sort order (e.g., "stars", "forks", "updated")
        """
        tool = f"search_{search_type}" if search_type != "repositories" else "search_repositories"
        params = {"query": query, "perPage": per_page}
        if sort:
            params["sort"] = sort
        self._logger.info(
            "github_search_starting",
            query=query,
            search_type=search_type,
            tool=tool,
        )
        result = await self.call_tool("github", tool, params)
        self._logger.info(
            "github_search_complete",
            success=result.success,
            error=result.error if not result.success else None,
        )
        return result

    async def search_huggingface(
        self,
        query: str,
        search_type: str = "models",
        limit: int = 20,
        sort: str | None = "downloads",
    ) -> MCPToolResult:
        """Convenience method for HuggingFace search.

        Args:
            query: Search query
            search_type: Type of search (models, datasets, papers, spaces)
            limit: Maximum results to return
            sort: Sort order ("trendingScore", "downloads", "likes")
        """
        # Map search types to actual HuggingFace MCP tool names
        # The HF MCP server at https://huggingface.co/mcp uses:
        #   hub_repo_search (for models/datasets), paper_search, space_search
        tool_map = {
            "models": "hub_repo_search",
            "datasets": "hub_repo_search",
            "papers": "paper_search",
            "spaces": "space_search",
        }
        tool = tool_map.get(search_type, "hub_repo_search")

        self._logger.info(
            "huggingface_search_starting",
            query=query,
            search_type=search_type,
            tool=tool,
        )
        params: dict[str, Any] = {"query": query}
        if tool == "hub_repo_search":
            params["limit"] = limit
            if sort:
                params["sort"] = sort
        elif tool == "paper_search":
            params["results_limit"] = limit
        elif tool == "space_search":
            params["limit"] = limit

        result = await self.call_tool(
            "hugging-face",  # MCP server name
            tool,
            params,
        )
        self._logger.info(
            "huggingface_search_complete",
            success=result.success,
            error=result.error if not result.success else None,
        )
        return result

    async def generate_image(
        self,
        prompt: str,
        resolution: str = "1024x1024 ( 1:1 )",
        steps: int = 8,
    ) -> MCPToolResult:
        """Generate image using HuggingFace Z-Image-Turbo.

        Args:
            prompt: Text prompt describing the desired image
            resolution: Output resolution (e.g., "1024x1024 ( 1:1 )")
            steps: Number of inference steps

        Returns:
            MCPToolResult with generated image data
        """
        self._logger.info(
            "image_generation_starting",
            prompt=prompt[:100],
            resolution=resolution,
        )
        result = await self.call_tool(
            "hugging-face",
            "gr1_z_image_turbo_generate",
            {
                "prompt": prompt,
                "resolution": resolution,
                "steps": steps,
                "random_seed": True,
            },
        )
        self._logger.info(
            "image_generation_complete",
            success=result.success,
            error=result.error if not result.success else None,
        )
        return result

    def _wrap_result(
        self,
        task: AgentTask,
        success: bool,
        data: Any = None,
        error: str | None = None,
        execution_time_ms: float = 0.0,
        **metadata: Any,
    ) -> AgentResult:
        """Create a standardized AgentResult."""
        return AgentResult(
            success=success,
            data=data,
            error=error,
            metadata=metadata,
            execution_time_ms=execution_time_ms,
            agent_name=self.name,
            task_id=task.id,
        )


class StrandsAgentMixin:
    """
    Mixin for agents using AWS Strands Agents SDK.

    Provides integration with Strands SDK for tool use and agent loops.
    """

    def _create_strands_agent(
        self,
        tools: list[Any] | None = None,
        model_id: str | None = None,
    ) -> Any:
        """
        Create a Strands Agent instance.

        Args:
            tools: List of tools to provide to the agent
            model_id: Bedrock model ID

        Returns:
            Strands Agent instance
        """
        try:
            import os as _os
            from strands import Agent
            from strands.models import BedrockModel

            resolved_model_id = model_id or _os.environ.get(
                "BEDROCK_MODEL_STANDARD", "global.anthropic.claude-sonnet-4-6"
            )
            model = BedrockModel(model_id=resolved_model_id)
            return Agent(model=model, tools=tools or [])
        except ImportError:
            logger.error("strands_sdk_not_installed")
            raise ImportError("strands-agents package is required")

    async def _run_strands_agent(
        self,
        agent: Any,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        Run a Strands agent with a prompt.

        Args:
            agent: Strands Agent instance
            prompt: User prompt
            context: Optional context to include

        Returns:
            Agent response
        """
        if context:
            prompt = f"Context: {context}\n\nTask: {prompt}"

        result = agent(prompt)
        return str(result)
