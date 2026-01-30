"""
Collaborative Reasoning Orchestrator for DOVA.

Unifies blackboard, ensemble, and iterative refinement patterns.
Proactively discovers and accesses tools based on task requirements.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

import structlog

from dova.services.blackboard import Blackboard, PostType
from dova.services.ensemble import EnsembleReasoning, AggregationMethod

logger = structlog.get_logger(__name__)


class CollaborationMode(Enum):
    BLACKBOARD = "blackboard"  # Shared workspace
    ENSEMBLE = "ensemble"  # Parallel reasoning
    ITERATIVE = "iterative"  # Refinement rounds
    HYBRID = "hybrid"  # Combination of all
    TOOL_AUGMENTED = "tool_augmented"  # Proactive tool discovery and use


@dataclass
class ToolExecutionResult:
    """Result from tool execution."""

    tool_name: str
    success: bool
    data: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0


@dataclass
class CollaborativeResult:
    """Result from collaborative reasoning."""

    final_answer: str
    confidence: float
    mode_used: CollaborationMode
    iterations: int = 0
    participating_agents: list[str] = field(default_factory=list)
    blackboard_synthesis: dict[str, Any] | None = None
    ensemble_result: Any | None = None
    refinement_history: list[dict[str, Any]] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    tool_results: list[ToolExecutionResult] = field(default_factory=list)
    tool_plan: Any | None = None


class CollaborativeReasoning:
    """
    Orchestrates multi-agent collaborative reasoning.

    Supports:
    - Blackboard: Agents post and build on shared insights
    - Ensemble: Multiple agents solve in parallel, synthesize
    - Iterative: Agents refine each other's work in rounds
    - Hybrid: Combines all patterns
    - Tool-Augmented: Proactively discovers and uses tools

    The orchestrator can proactively:
    - Analyze tasks to determine required capabilities
    - Discover available tools from MCP, sandbox, and internal services
    - Select optimal tools based on task requirements and configuration
    - Execute tools as part of the reasoning process
    """

    def __init__(
        self,
        llm_func: Any = None,
        settings: Any = None,
        mcp_client: Any = None,
        sandbox_executor: Any = None,
        memory_service: Any = None,
    ):
        self.llm_func = llm_func
        self.settings = settings
        self.mcp_client = mcp_client
        self.sandbox_executor = sandbox_executor
        self.memory_service = memory_service
        self.blackboard = Blackboard()
        self.ensemble = EnsembleReasoning(llm_func)
        self._tool_resolver: Any = None
        self._logger = logger.bind(service="collaborative")

    @property
    def tool_resolver(self) -> Any:
        """Lazy initialization of tool resolver."""
        if self._tool_resolver is None:
            from dova.services.tool_resolver import ToolResolver

            self._tool_resolver = ToolResolver(
                settings=self.settings,
                sandbox_executor=self.sandbox_executor,
                memory_service=self.memory_service,
            )
            self._tool_resolver.discover_tools()
        return self._tool_resolver

    async def reason(
        self,
        problem: str,
        agents: list[Any],
        mode: CollaborationMode = CollaborationMode.HYBRID,
        max_iterations: int = 3,
        context: dict[str, Any] | None = None,
        use_tools: bool = True,
    ) -> CollaborativeResult:
        """
        Execute collaborative reasoning.

        Args:
            problem: The problem to solve
            agents: List of participating agents
            mode: Collaboration mode to use
            max_iterations: Max refinement rounds (for iterative/hybrid)
            context: Shared context
            use_tools: Whether to proactively discover and use tools

        Returns:
            CollaborativeResult with final answer
        """
        self._logger.info(
            "collaborative_start",
            problem=problem[:100],
            mode=mode.value,
            agents=[getattr(a, "name", str(a)) for a in agents],
            use_tools=use_tools,
        )

        # For tool-augmented mode, analyze and execute with tools first
        if mode == CollaborationMode.TOOL_AUGMENTED:
            return await self._tool_augmented_reasoning(
                problem, agents, max_iterations, context
            )

        # For other modes, optionally augment with tools
        tool_context = context or {}
        if use_tools:
            tool_results = await self._gather_tool_context(problem, context)
            if tool_results:
                tool_context = {**(context or {}), "tool_results": tool_results}

        if mode == CollaborationMode.BLACKBOARD:
            result = await self._blackboard_reasoning(problem, agents, tool_context)
        elif mode == CollaborationMode.ENSEMBLE:
            result = await self._ensemble_reasoning(problem, agents, tool_context)
        elif mode == CollaborationMode.ITERATIVE:
            result = await self._iterative_reasoning(
                problem, agents, max_iterations, tool_context
            )
        else:  # HYBRID
            result = await self._hybrid_reasoning(
                problem, agents, max_iterations, tool_context
            )

        # Add tool information to result
        if use_tools and tool_context.get("tool_results"):
            result.tools_used = [tr["tool"] for tr in tool_context["tool_results"]]
            result.tool_results = [
                ToolExecutionResult(
                    tool_name=tr["tool"],
                    success=tr.get("success", True),
                    data=tr.get("data"),
                )
                for tr in tool_context["tool_results"]
            ]

        return result

    async def _gather_tool_context(
        self,
        problem: str,
        context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """
        Proactively gather relevant context using tools.

        Analyzes the problem and executes appropriate tools to gather
        information that will help with reasoning.
        """
        try:
            plan = self.tool_resolver.create_plan(problem, context)

            if not plan.selected_tools:
                return []

            # Execute high-priority search tools in parallel
            from dova.services.tool_resolver import ToolCategory

            search_tools = [
                t for t in plan.selected_tools if t.category == ToolCategory.SEARCH
            ][:3]

            if not search_tools:
                return []

            results = await self._execute_tools_parallel(search_tools, problem, context)
            return results

        except Exception as e:
            self._logger.warning("tool_context_gathering_failed", error=str(e))
            return []

    async def _execute_tools_parallel(
        self,
        tools: list[Any],
        problem: str,
        context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Execute multiple tools in parallel."""
        import time

        async def execute_tool(tool: Any) -> dict[str, Any]:
            start = time.time()
            try:
                result = await self._execute_single_tool(tool, problem, context)
                return {
                    "tool": tool.name,
                    "success": True,
                    "data": result,
                    "time_ms": (time.time() - start) * 1000,
                }
            except Exception as e:
                self._logger.warning(
                    "tool_execution_failed",
                    tool=tool.name,
                    error=str(e),
                )
                return {
                    "tool": tool.name,
                    "success": False,
                    "error": str(e),
                    "time_ms": (time.time() - start) * 1000,
                }

        results = await asyncio.gather(*[execute_tool(t) for t in tools])
        return [r for r in results if r.get("success")]

    async def _execute_single_tool(
        self,
        tool: Any,
        problem: str,
        context: dict[str, Any] | None,
    ) -> Any:
        """Execute a single tool based on its type."""
        if tool.source == "mcp" and self.mcp_client:
            # Parse MCP tool name: "mcp:server:tool_name"
            parts = tool.name.split(":")
            if len(parts) >= 3:
                server = parts[1]
                tool_name = parts[2]

                # Extract query from problem
                query = self._extract_query(problem)

                return await self.mcp_client.invoke(
                    server=server,
                    tool=tool_name,
                    params={"query": query, "max_results": 5},
                )

        elif tool.source == "sandbox" and self.sandbox_executor:
            # For sandbox tools, we don't execute proactively
            # (would need explicit code)
            return None

        elif tool.source == "internal":
            # Handle internal tools
            if "memory:recall" in tool.name and self.memory_service:
                query = self._extract_query(problem)
                return await self.memory_service.search(query)

        return None

    def _extract_query(self, problem: str) -> str:
        """Extract a search query from the problem statement."""
        # Simple extraction - take key terms
        import re

        # Remove common question words
        clean = re.sub(
            r"\b(what|how|why|when|where|which|can|could|would|should|is|are|the)\b",
            "",
            problem.lower(),
        )

        # Take first 100 chars or until first sentence break
        query = clean.strip()[:100]
        if "." in query:
            query = query.split(".")[0]

        return query.strip() or problem[:50]

    async def _blackboard_reasoning(
        self,
        problem: str,
        agents: list[Any],
        context: dict[str, Any] | None,
    ) -> CollaborativeResult:
        """Blackboard-based collaborative reasoning."""
        self.blackboard.clear()

        # Phase 1: Each agent posts initial hypothesis
        for agent in agents:
            if hasattr(agent, "reason"):
                trace = await agent.reason(problem, context=context, max_iterations=2)
                await self.blackboard.post(
                    agent_name=agent.name,
                    post_type=PostType.HYPOTHESIS,
                    content=trace.final_answer,
                    confidence=trace.confidence,
                )
            elif hasattr(agent, "think"):
                answer = await agent.think(f"Propose a hypothesis for: {problem}")
                await self.blackboard.post(
                    agent_name=getattr(agent, "name", "unknown"),
                    post_type=PostType.HYPOTHESIS,
                    content=answer,
                    confidence=0.5,
                )

        # Phase 2: Agents review and add evidence/refinements
        for agent in agents:
            posts = await self.blackboard.get_context(
                agent_name=agent.name,
                exclude_own=True,
                post_types=[PostType.HYPOTHESIS],
            )

            for post in posts[:3]:
                if hasattr(agent, "think"):
                    review = await agent.think(
                        f"Review this hypothesis and provide supporting or refuting evidence:\n{post.content}"
                    )
                    await self.blackboard.post(
                        agent_name=agent.name,
                        post_type=PostType.EVIDENCE,
                        content=review,
                        references=[post.id],
                    )

        # Phase 3: Synthesize
        synthesis = await self.blackboard.synthesize()

        # Get best hypothesis
        best = (
            synthesis["hypotheses"][0]
            if synthesis["hypotheses"]
            else {"content": "", "confidence": 0}
        )

        return CollaborativeResult(
            final_answer=best["content"],
            confidence=best.get("confidence", 0.5),
            mode_used=CollaborationMode.BLACKBOARD,
            participating_agents=[getattr(a, "name", str(a)) for a in agents],
            blackboard_synthesis=synthesis,
        )

    async def _ensemble_reasoning(
        self,
        problem: str,
        agents: list[Any],
        context: dict[str, Any] | None,
    ) -> CollaborativeResult:
        """Ensemble-based collaborative reasoning."""
        result = await self.ensemble.reason(
            problem=problem,
            agents=agents,
            method=AggregationMethod.SYNTHESIS,
            context=context,
        )

        return CollaborativeResult(
            final_answer=result.synthesized_answer,
            confidence=result.confidence,
            mode_used=CollaborationMode.ENSEMBLE,
            participating_agents=[a.agent_name for a in result.individual_answers],
            ensemble_result=result,
        )

    async def _iterative_reasoning(
        self,
        problem: str,
        agents: list[Any],
        max_iterations: int,
        context: dict[str, Any] | None,
    ) -> CollaborativeResult:
        """Iterative refinement-based reasoning."""
        if not agents:
            return CollaborativeResult(
                final_answer="",
                confidence=0.0,
                mode_used=CollaborationMode.ITERATIVE,
            )

        # Start with first agent's answer
        current_answer = ""
        current_confidence = 0.5
        if hasattr(agents[0], "reason"):
            trace = await agents[0].reason(problem, context=context)
            current_answer = trace.refined_answer or trace.final_answer
            current_confidence = trace.confidence
        elif hasattr(agents[0], "think"):
            current_answer = await agents[0].think(f"Answer: {problem}")
            current_confidence = 0.5

        history = [{"agent": getattr(agents[0], "name", "0"), "answer": current_answer}]

        # Iterate through agents for refinement
        for iteration in range(1, max_iterations):
            agent_idx = iteration % len(agents)
            agent = agents[agent_idx]

            if hasattr(agent, "reflect"):
                refined, critique = await agent.reflect(current_answer, problem)
                current_answer = refined
                history.append(
                    {
                        "agent": getattr(agent, "name", str(agent_idx)),
                        "critique": critique,
                        "refined": refined,
                    }
                )
            elif hasattr(agent, "think"):
                prompt = f"""Review and improve this answer:

Problem: {problem}
Current Answer: {current_answer}

Provide an improved version."""
                current_answer = await agent.think(prompt)
                history.append(
                    {
                        "agent": getattr(agent, "name", str(agent_idx)),
                        "refined": current_answer,
                    }
                )

        return CollaborativeResult(
            final_answer=current_answer,
            confidence=current_confidence,
            mode_used=CollaborationMode.ITERATIVE,
            iterations=max_iterations,
            participating_agents=[getattr(a, "name", str(a)) for a in agents],
            refinement_history=history,
        )

    async def _hybrid_reasoning(
        self,
        problem: str,
        agents: list[Any],
        max_iterations: int,
        context: dict[str, Any] | None,
    ) -> CollaborativeResult:
        """Hybrid: Ensemble → Blackboard → Iterative refinement."""
        # Step 1: Ensemble for initial diverse answers
        ensemble_result = await self._ensemble_reasoning(problem, agents, context)

        # Step 2: Post to blackboard for evidence gathering
        self.blackboard.clear()
        await self.blackboard.post(
            agent_name="ensemble",
            post_type=PostType.HYPOTHESIS,
            content=ensemble_result.final_answer,
            confidence=ensemble_result.confidence,
        )

        # Add dissenting views
        if ensemble_result.ensemble_result:
            for view in ensemble_result.ensemble_result.dissenting_views:
                await self.blackboard.post(
                    agent_name="dissent",
                    post_type=PostType.EVIDENCE,
                    content=view,
                    confidence=0.3,
                )

        # Step 3: Iterative refinement on ensemble answer
        iterative_result = await self._iterative_reasoning(
            problem=f"Refine this answer: {ensemble_result.final_answer}\nOriginal problem: {problem}",
            agents=agents[:2],  # Use subset for refinement
            max_iterations=min(2, max_iterations),
            context=context,
        )

        return CollaborativeResult(
            final_answer=iterative_result.final_answer,
            confidence=(ensemble_result.confidence + iterative_result.confidence) / 2,
            mode_used=CollaborationMode.HYBRID,
            iterations=max_iterations,
            participating_agents=list(
                set(
                    ensemble_result.participating_agents
                    + iterative_result.participating_agents
                )
            ),
            blackboard_synthesis=await self.blackboard.synthesize(),
            ensemble_result=ensemble_result.ensemble_result,
            refinement_history=iterative_result.refinement_history,
        )

    async def _tool_augmented_reasoning(
        self,
        problem: str,
        agents: list[Any],
        max_iterations: int,
        context: dict[str, Any] | None,
    ) -> CollaborativeResult:
        """
        Tool-augmented reasoning mode.

        Proactively discovers and uses tools based on task analysis,
        then integrates tool results with agent reasoning.
        """
        tool_results: list[ToolExecutionResult] = []
        tools_used: list[str] = []

        # Step 1: Analyze task and create tool plan
        plan = self.tool_resolver.create_plan(problem, context)

        self._logger.info(
            "tool_plan_created",
            categories=[c.value for c in plan.requirements.categories],
            selected_tools=[t.name for t in plan.selected_tools],
        )

        # Step 2: Execute tools in planned order
        if plan.selected_tools:
            results = await self._execute_tools_parallel(
                plan.selected_tools[:5],  # Limit to top 5 tools
                problem,
                context,
            )

            for r in results:
                tools_used.append(r["tool"])
                tool_results.append(
                    ToolExecutionResult(
                        tool_name=r["tool"],
                        success=r.get("success", False),
                        data=r.get("data"),
                        error=r.get("error"),
                        execution_time_ms=r.get("time_ms", 0),
                    )
                )

        # Step 3: Build enriched context from tool results
        enriched_context = {**(context or {})}
        if tool_results:
            enriched_context["tool_data"] = {
                tr.tool_name: tr.data
                for tr in tool_results
                if tr.success and tr.data
            }

        # Step 4: Run hybrid reasoning with enriched context
        if agents:
            hybrid_result = await self._hybrid_reasoning(
                problem, agents, max_iterations, enriched_context
            )
            final_answer = hybrid_result.final_answer
            confidence = hybrid_result.confidence
            participating = hybrid_result.participating_agents
            synthesis = hybrid_result.blackboard_synthesis
            ensemble = hybrid_result.ensemble_result
            history = hybrid_result.refinement_history
        else:
            # No agents - synthesize from tool results alone
            final_answer = self._synthesize_tool_results(problem, tool_results)
            confidence = 0.6 if tool_results else 0.3
            participating = []
            synthesis = None
            ensemble = None
            history = []

        return CollaborativeResult(
            final_answer=final_answer,
            confidence=confidence,
            mode_used=CollaborationMode.TOOL_AUGMENTED,
            iterations=max_iterations,
            participating_agents=participating,
            blackboard_synthesis=synthesis,
            ensemble_result=ensemble,
            refinement_history=history,
            tools_used=tools_used,
            tool_results=tool_results,
            tool_plan=plan,
        )

    def _synthesize_tool_results(
        self,
        problem: str,
        tool_results: list[ToolExecutionResult],
    ) -> str:
        """Synthesize an answer from tool results when no agents available."""
        if not tool_results:
            return f"Unable to find relevant information for: {problem}"

        successful = [tr for tr in tool_results if tr.success and tr.data]
        if not successful:
            return f"Tools executed but returned no useful data for: {problem}"

        # Build summary from tool data
        parts = [f"Based on analysis of '{problem}':"]
        for tr in successful:
            if isinstance(tr.data, dict):
                summary = tr.data.get("summary", str(tr.data)[:200])
            elif isinstance(tr.data, list):
                summary = f"Found {len(tr.data)} results"
            else:
                summary = str(tr.data)[:200]
            parts.append(f"- {tr.tool_name}: {summary}")

        return "\n".join(parts)
