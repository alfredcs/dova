"""
Base Agent Implementation for DOVA.

Provides the foundation for all DOVA agents with unified interfaces
for LLM calls, MCP tool invocation, and lifecycle management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
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

        self._logger.debug(
            "llm_call_complete",
            provider=response.provider,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        return response.content

    async def call_tool(
        self,
        server: str,
        tool: str,
        params: dict[str, Any],
    ) -> MCPToolResult:
        """
        Invoke an MCP tool.

        Args:
            server: MCP server name (e.g., "arxiv", "github")
            tool: Tool name to invoke
            params: Tool parameters

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
                result = await self.mcp_client.invoke(server, tool, params)

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
    ) -> MCPToolResult:
        """Convenience method for ArXiv search."""
        return await self.call_tool(
            "arxiv",
            "search_arxiv",  # Fixed: was "search_papers"
            {"request": {"query": query, "max_results": max_results}},
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
    ) -> MCPToolResult:
        """Convenience method for HuggingFace search.

        Args:
            query: Search query
            search_type: Type of search (models, datasets, papers, spaces)
            limit: Maximum results to return
        """
        tool = f"{search_type[:-1]}_search" if search_type.endswith("s") else f"{search_type}_search"
        self._logger.info(
            "huggingface_search_starting",
            query=query,
            search_type=search_type,
            tool=tool,
        )
        result = await self.call_tool(
            "hugging-face",  # Fixed: MCP server name uses hyphen
            tool,
            {"query": query, "limit": limit},
        )
        self._logger.info(
            "huggingface_search_complete",
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
        model_id: str = "anthropic.claude-sonnet-4-20250514-v1:0",
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
            from strands import Agent
            from strands.models import BedrockModel

            model = BedrockModel(model_id=model_id)
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
