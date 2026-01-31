"""
Master Orchestrator Agent for DOVA.

The orchestrator is responsible for:
- Understanding user intent
- Decomposing tasks into sub-tasks
- Dispatching to specialized agents
- Aggregating and synthesizing results
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from dova.agents.base import AgentResult, AgentTask, BaseAgent
from dova.config.providers import LLMRouter, TaskType
from dova.services.collaborative import CollaborativeReasoning, CollaborationMode
from dova.utils.metrics import MetricsCollector

logger = structlog.get_logger(__name__)


class UserIntent(Enum):
    """Classified user intents."""

    RESEARCH_QUERY = "research_query"
    CODE_SEARCH = "code_search"
    PAPER_SEARCH = "paper_search"
    MODEL_SEARCH = "model_search"
    INNOVATION_REQUEST = "innovation_request"
    VALIDATION_REQUEST = "validation_request"
    PROFILE_UPDATE = "profile_update"
    GENERAL_QUESTION = "general_question"


class ReasoningMode(Enum):
    """Reasoning depth modes for orchestration."""

    QUICK = "quick"  # Single-pass, no reflection
    STANDARD = "standard"  # ReAct + self-reflection
    DEEP = "deep"  # Full collaborative reasoning
    COLLABORATIVE = "collaborative"  # Blackboard + ensemble


@dataclass
class ParsedIntent:
    """Parsed user intent with extracted entities."""

    intent: UserIntent
    confidence: float
    entities: dict[str, Any] = field(default_factory=dict)
    requires_profiling: bool = False
    requires_validation: bool = False


@dataclass
class TaskNode:
    """A node in the task execution graph."""

    id: str
    agent_type: str
    task: AgentTask
    dependencies: list[str] = field(default_factory=list)
    result: AgentResult | None = None
    status: str = "pending"


class DOVAOrchestrator(BaseAgent):
    """
    Master Orchestrator Agent for DOVA platform.

    Coordinates all specialized agents and manages the research workflow.
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        agents: dict[str, BaseAgent] | None = None,
        mcp_client: Any | None = None,
        metrics: MetricsCollector | None = None,
        reasoning_mode: ReasoningMode = ReasoningMode.STANDARD,
    ):
        super().__init__(llm_router, mcp_client, metrics)
        self.agents = agents or {}
        self._task_graph: dict[str, TaskNode] = {}
        self.reasoning_mode = reasoning_mode
        self._collaborative = CollaborativeReasoning(llm_func=self.think)

    @property
    def system_prompt(self) -> str:
        return """You are the DOVA Master Orchestrator, an AI agent responsible for understanding user research queries and coordinating specialized agents.

Your capabilities:
1. Parse user intent and extract relevant entities (topics, authors, time ranges, etc.)
2. Determine which specialized agents to invoke (research, profiling, validation, synthesis)
3. Plan efficient execution strategies (parallel vs sequential)
4. Synthesize results from multiple sources

When analyzing a user query, identify:
- Primary intent (research, code search, paper search, etc.)
- Key entities (topics, technologies, authors, repositories)
- Constraints (time range, language, quality requirements)
- Whether personalization (user profile) would improve results
- Whether validation is needed for any code/implementations

Respond in a structured JSON format."""

    async def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute the orchestration workflow.

        Args:
            task: Task containing the user query

        Returns:
            AgentResult with synthesized research results
        """
        start_time = time.time()

        try:
            # Step 1: Parse user intent
            query = task.params.get("query", "")
            if not query:
                return self._wrap_result(task, False, error="No query provided")

            self._logger.info("orchestrator_starting", query=query, user_id=task.user_id)

            intent = await self._classify_intent(query)
            self._logger.debug("intent_classified", intent=intent.intent.value, confidence=intent.confidence)

            # Step 2: Build task graph based on intent
            task_graph = await self._build_task_graph(query, intent, task)

            # Step 3: Execute task graph
            results = await self._execute_task_graph(task_graph)

            # Step 4: Synthesize results
            synthesized = await self._synthesize_results(query, intent, results)

            execution_time = (time.time() - start_time) * 1000
            return self._wrap_result(
                task,
                True,
                data=synthesized,
                execution_time_ms=execution_time,
                intent=intent.intent.value,
                tasks_executed=len(results),
            )

        except Exception as e:
            self._logger.exception("orchestrator_error", error=str(e))
            return self._wrap_result(
                task,
                False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _classify_intent(self, query: str) -> ParsedIntent:
        """Classify user intent from query."""
        classification_prompt = f"""Analyze this research query and classify the user's intent.

Query: "{query}"

Respond with JSON:
{{
    "intent": "<one of: research_query, code_search, paper_search, model_search, innovation_request, validation_request, general_question>",
    "confidence": <0.0-1.0>,
    "entities": {{
        "topics": ["<extracted topics>"],
        "technologies": ["<technologies mentioned>"],
        "authors": ["<author names if mentioned>"],
        "repositories": ["<repo names if mentioned>"],
        "time_range": "<if specified, e.g., 'last 6 months'>",
        "constraints": ["<other constraints>"]
    }},
    "requires_profiling": <true if personalization would help>,
    "requires_validation": <true if code validation is requested>
}}"""

        response = await self.think(
            classification_prompt,
            task_type=TaskType.CLASSIFICATION,
            temperature=0.3,
        )

        # Parse JSON response
        import json

        try:
            # Handle potential markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            parsed = json.loads(response.strip())
            return ParsedIntent(
                intent=UserIntent(parsed.get("intent", "general_question")),
                confidence=parsed.get("confidence", 0.5),
                entities=parsed.get("entities", {}),
                requires_profiling=parsed.get("requires_profiling", False),
                requires_validation=parsed.get("requires_validation", False),
            )
        except (json.JSONDecodeError, ValueError) as e:
            self._logger.warning("intent_parse_error", error=str(e), response=response)
            return ParsedIntent(
                intent=UserIntent.RESEARCH_QUERY,
                confidence=0.5,
                entities={"topics": [query]},
            )

    async def _build_task_graph(
        self,
        query: str,
        intent: ParsedIntent,
        parent_task: AgentTask,
    ) -> dict[str, TaskNode]:
        """Build task execution graph based on intent."""
        graph: dict[str, TaskNode] = {}

        # Get requested sources from task params (defaults to all if not specified)
        requested_sources = parent_task.params.get("sources", ["arxiv", "github", "huggingface"])

        # Filter to only configured sources (check MCP registry)
        available_sources = []
        if self.mcp_client:
            for source in requested_sources:
                server = self.mcp_client.registry.get_server(source)
                if server:
                    available_sources.append(source)
        else:
            available_sources = requested_sources

        # Always add research tasks based on intent
        if intent.intent in [
            UserIntent.RESEARCH_QUERY,
            UserIntent.PAPER_SEARCH,
            UserIntent.CODE_SEARCH,
            UserIntent.MODEL_SEARCH,
        ]:
            # ArXiv search (only if configured)
            if "arxiv" in available_sources:
                graph["arxiv_search"] = TaskNode(
                    id="arxiv_search",
                    agent_type="research",
                    task=AgentTask(
                        type="search",
                        params={
                            "source": "arxiv",
                            "query": query,
                            "entities": intent.entities,
                        },
                        user_id=parent_task.user_id,
                    ),
                )

            # GitHub search (only if configured)
            if "github" in available_sources:
                graph["github_search"] = TaskNode(
                    id="github_search",
                    agent_type="research",
                    task=AgentTask(
                        type="search",
                        params={
                            "source": "github",
                            "query": query,
                            "entities": intent.entities,
                        },
                        user_id=parent_task.user_id,
                    ),
                )

            # HuggingFace search (only if configured)
            if "huggingface" in available_sources:
                graph["hf_search"] = TaskNode(
                    id="hf_search",
                    agent_type="research",
                    task=AgentTask(
                        type="search",
                        params={
                            "source": "huggingface",
                            "query": query,
                            "entities": intent.entities,
                        },
                        user_id=parent_task.user_id,
                    ),
                )

        # Add profiling if needed
        if intent.requires_profiling and parent_task.user_id:
            graph["profiling"] = TaskNode(
                id="profiling",
                agent_type="profiling",
                task=AgentTask(
                    type="get_preferences",
                    params={"user_id": parent_task.user_id},
                    user_id=parent_task.user_id,
                ),
            )

        # Add synthesis task (depends on search results)
        search_deps = [k for k in graph.keys() if k.endswith("_search")]
        if search_deps:
            graph["synthesis"] = TaskNode(
                id="synthesis",
                agent_type="synthesis",
                task=AgentTask(
                    type="synthesize",
                    params={"query": query, "intent": intent.intent.value},
                    user_id=parent_task.user_id,
                ),
                dependencies=search_deps,
            )

        return graph

    async def _execute_task_graph(
        self,
        graph: dict[str, TaskNode],
    ) -> dict[str, AgentResult]:
        """Execute task graph with dependency management."""
        results: dict[str, AgentResult] = {}
        completed: set[str] = set()

        while len(completed) < len(graph):
            # Find tasks ready to execute (no pending dependencies)
            ready = [
                node_id
                for node_id, node in graph.items()
                if node_id not in completed
                and all(dep in completed for dep in node.dependencies)
            ]

            if not ready:
                self._logger.warning("task_graph_deadlock", completed=list(completed))
                break

            # Execute ready tasks in parallel
            tasks = []
            for node_id in ready:
                node = graph[node_id]
                agent = self.agents.get(node.agent_type)
                if agent:
                    # Pass dependency results to task context
                    node.task.context["dependency_results"] = {
                        dep: results.get(dep) for dep in node.dependencies
                    }
                    tasks.append(self._execute_single_task(node_id, agent, node.task))
                else:
                    self._logger.warning("agent_not_found", agent_type=node.agent_type)
                    completed.add(node_id)

            if tasks:
                task_results = await asyncio.gather(*tasks, return_exceptions=True)
                for (node_id, result) in task_results:
                    if isinstance(result, Exception):
                        results[node_id] = AgentResult(
                            success=False,
                            error=str(result),
                            agent_name=graph[node_id].agent_type,
                        )
                    else:
                        results[node_id] = result
                    completed.add(node_id)

        return results

    async def _execute_single_task(
        self,
        node_id: str,
        agent: BaseAgent,
        task: AgentTask,
    ) -> tuple[str, AgentResult]:
        """Execute a single task and return with its node ID."""
        self._logger.debug("executing_task", node_id=node_id, agent=agent.name)
        try:
            result = await agent.execute(task)
            return (node_id, result)
        except Exception as e:
            self._logger.exception("task_execution_error", node_id=node_id, error=str(e))
            return (node_id, AgentResult(success=False, error=str(e), agent_name=agent.name))

    async def _synthesize_results(
        self,
        query: str,
        intent: ParsedIntent,
        results: dict[str, AgentResult],
    ) -> dict[str, Any]:
        """Synthesize results from all agents into a coherent response."""
        # Collect raw search results
        raw_results = self._collect_raw_results(results)

        # If we have a synthesis agent result, merge it with raw results
        if "synthesis" in results and results["synthesis"].success:
            synthesis_data = results["synthesis"].data
            # Merge raw results into synthesis output
            return {
                "summary": synthesis_data.get("executive_summary", ""),
                "papers": raw_results.get("papers", []),
                "repositories": raw_results.get("repositories", []),
                "models": raw_results.get("models", []),
                "datasets": raw_results.get("datasets", []),
                "insights": synthesis_data.get("key_findings", []),
                "recommendations": synthesis_data.get("recommendations", []),
                "knowledge_gaps": synthesis_data.get("knowledge_gaps", []),
                "confidence_score": synthesis_data.get("confidence_score", 0.5),
            }

    def _collect_raw_results(self, results: dict[str, AgentResult]) -> dict[str, list]:
        """Collect raw search results from agent results."""
        collected = {
            "papers": [],
            "repositories": [],
            "models": [],
            "datasets": [],
        }

        for node_id, result in results.items():
            if not result.success or not result.data:
                continue

            data = result.data
            if hasattr(data, "papers"):
                # ResearchFindings object
                collected["papers"].extend(
                    [{"title": p.title, "url": p.url, "description": p.description, **p.metadata}
                     for p in data.papers] if hasattr(data, "papers") else []
                )
                collected["repositories"].extend(
                    [{"name": r.title, "url": r.url, "description": r.description, **r.metadata}
                     for r in data.repositories] if hasattr(data, "repositories") else []
                )
                collected["models"].extend(
                    [{"id": m.title, "url": m.url, "description": m.description, **m.metadata}
                     for m in data.models] if hasattr(data, "models") else []
                )
                collected["datasets"].extend(
                    [{"id": d.title, "url": d.url, "description": d.description, **d.metadata}
                     for d in data.datasets] if hasattr(data, "datasets") else []
                )
            elif isinstance(data, dict):
                # Dict-based results
                for key in ["papers", "repositories", "models", "datasets"]:
                    if key in data and isinstance(data[key], list):
                        collected[key].extend(data[key])

        return collected

    def _format_results_for_synthesis(self, results: dict[str, AgentResult]) -> str:
        """Format agent results for the synthesis prompt."""
        formatted = []
        for node_id, result in results.items():
            if result.success and result.data:
                formatted.append(f"\n## {node_id}:\n{result.data}")
            elif not result.success:
                formatted.append(f"\n## {node_id}: Error - {result.error}")
        return "\n".join(formatted) if formatted else "No results available"

    def register_agent(self, agent_type: str, agent: BaseAgent) -> None:
        """Register a specialized agent."""
        self.agents[agent_type] = agent
        self._logger.info("agent_registered", agent_type=agent_type, agent_name=agent.name)

    async def _execute_collaborative(
        self,
        query: str,
        intent: ParsedIntent,
        parent_task: AgentTask,
    ) -> dict[str, Any]:
        """Execute with full collaborative reasoning."""
        agents_list = list(self.agents.values())

        if self.reasoning_mode == ReasoningMode.COLLABORATIVE:
            mode = CollaborationMode.HYBRID
        elif self.reasoning_mode == ReasoningMode.DEEP:
            mode = CollaborationMode.ENSEMBLE
        else:
            mode = CollaborationMode.BLACKBOARD

        result = await self._collaborative.reason(
            problem=query,
            agents=agents_list,
            mode=mode,
            context={"intent": intent.intent.value, "entities": intent.entities},
        )

        return {
            "answer": result.final_answer,
            "confidence": result.confidence,
            "mode": result.mode_used.value,
            "participants": result.participating_agents,
        }
