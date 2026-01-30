"""
Reasoning Mixin for DOVA Agents.

Provides ReAct-style reasoning, self-reflection, and working memory.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class StepType(Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    REFLECTION = "reflection"


@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""

    step_type: StepType
    content: str
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningTrace:
    """Complete trace of agent's reasoning process."""

    problem: str
    steps: list[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    total_iterations: int = 0
    self_critique: str | None = None
    refined_answer: str | None = None
    confidence: float = 0.0


class ReasoningMixin:
    """
    Mixin providing advanced reasoning capabilities to agents.

    Requires the host class to have:
    - think(prompt, ...) -> str method for LLM calls
    - _logger for logging
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._scratchpad: dict[str, Any] = {}
        self._current_trace: ReasoningTrace | None = None

    async def reason(
        self,
        problem: str,
        context: dict[str, Any] | None = None,
        max_iterations: int = 5,
        reflect: bool = True,
        available_actions: list[str] | None = None,
    ) -> ReasoningTrace:
        """
        ReAct-style reasoning loop: Thought → Action → Observation → (repeat)

        Args:
            problem: The problem to reason about
            context: Additional context for reasoning
            max_iterations: Maximum think-act-observe cycles
            reflect: Whether to self-critique before finalizing
            available_actions: List of actions the agent can take

        Returns:
            ReasoningTrace with full reasoning history
        """
        trace = ReasoningTrace(problem=problem)
        self._current_trace = trace
        self._scratchpad = {"context": context or {}, "observations": []}

        actions = available_actions or ["search", "analyze", "synthesize", "conclude"]

        for iteration in range(max_iterations):
            # THOUGHT: Reason about what to do next
            thought = await self._think_step(problem, trace, actions)
            trace.steps.append(
                ReasoningStep(
                    step_type=StepType.THOUGHT,
                    content=thought["reasoning"],
                    confidence=thought.get("confidence", 0.5),
                )
            )

            # Check if agent wants to conclude
            if thought.get("action") == "conclude":
                trace.final_answer = thought.get("conclusion", "")
                trace.confidence = thought.get("confidence", 0.5)
                break

            # ACTION: Execute the chosen action
            action_result = await self._action_step(
                thought["action"],
                thought.get("action_input", {}),
            )
            trace.steps.append(
                ReasoningStep(
                    step_type=StepType.ACTION,
                    content=f"{thought['action']}: {thought.get('action_input', {})}",
                    metadata={
                        "action": thought["action"],
                        "input": thought.get("action_input"),
                    },
                )
            )

            # OBSERVATION: Record what was learned
            trace.steps.append(
                ReasoningStep(
                    step_type=StepType.OBSERVATION,
                    content=str(action_result),
                    metadata={"result": action_result},
                )
            )
            self._scratchpad["observations"].append(action_result)

        trace.total_iterations = len(
            [s for s in trace.steps if s.step_type == StepType.THOUGHT]
        )

        # REFLECTION: Self-critique and refine if enabled
        if reflect and trace.final_answer:
            refined, critique = await self.reflect(trace.final_answer, problem)
            trace.self_critique = critique
            trace.refined_answer = refined
            trace.steps.append(
                ReasoningStep(
                    step_type=StepType.REFLECTION,
                    content=critique,
                    metadata={"original": trace.final_answer, "refined": refined},
                )
            )

        self._current_trace = None
        return trace

    async def _think_step(
        self,
        problem: str,
        trace: ReasoningTrace,
        actions: list[str],
    ) -> dict[str, Any]:
        """Generate a thought about what to do next."""
        history = self._format_trace_history(trace)

        prompt = f"""You are reasoning step-by-step about a problem.

Problem: {problem}

Previous reasoning:
{history}

Scratchpad (working memory):
{self._format_scratchpad()}

Available actions: {actions}

Think about what you know, what you need to find out, and what action to take next.
If you have enough information, choose "conclude" as your action.

Respond in JSON:
{{
    "reasoning": "<your thought process>",
    "action": "<one of: {', '.join(actions)}>",
    "action_input": {{"<relevant parameters>"}},
    "conclusion": "<if action is conclude, your final answer>",
    "confidence": <0.0-1.0>
}}"""

        response = await self.think(prompt, temperature=0.3)
        return self._parse_json_response(response)

    async def _action_step(
        self,
        action: str,
        action_input: dict[str, Any],
    ) -> Any:
        """Execute an action. Override in subclasses for specific actions."""
        # Default implementation - subclasses should override
        if action == "search" and hasattr(self, "search_arxiv"):
            query = action_input.get("query", "")
            result = await self.search_arxiv(query)
            return result.data if result.success else result.error
        elif action == "analyze":
            return f"Analyzed: {action_input}"
        elif action == "synthesize":
            return f"Synthesized: {action_input}"
        return f"Action {action} not implemented"

    async def reflect(
        self,
        draft_answer: str,
        original_problem: str,
    ) -> tuple[str, str]:
        """
        Self-critique and refine an answer.

        Args:
            draft_answer: The initial answer to critique
            original_problem: The original problem for context

        Returns:
            Tuple of (refined_answer, critique)
        """
        prompt = f"""Review and critique this answer, then provide an improved version.

Original Problem: {original_problem}

Draft Answer: {draft_answer}

Critique the answer for:
1. Accuracy - Are there factual errors?
2. Completeness - Is anything important missing?
3. Clarity - Is it easy to understand?
4. Logic - Does the reasoning flow well?

Respond in JSON:
{{
    "critique": "<specific issues found>",
    "improvements": ["<improvement 1>", "<improvement 2>"],
    "refined_answer": "<the improved answer>",
    "confidence_change": <-1.0 to 1.0, how much confidence changed>
}}"""

        response = await self.think(prompt, temperature=0.3)
        parsed = self._parse_json_response(response)

        return parsed.get("refined_answer", draft_answer), parsed.get(
            "critique", "No issues found"
        )

    def get_scratchpad(self) -> dict[str, Any]:
        """Get current working memory."""
        return self._scratchpad.copy()

    def update_scratchpad(self, key: str, value: Any) -> None:
        """Update working memory."""
        self._scratchpad[key] = value

    def _format_trace_history(self, trace: ReasoningTrace) -> str:
        """Format reasoning trace for context."""
        if not trace.steps:
            return "No previous reasoning steps."

        lines = []
        for i, step in enumerate(trace.steps):
            lines.append(
                f"{i+1}. [{step.step_type.value.upper()}] {step.content[:200]}..."
            )
        return "\n".join(lines)

    def _format_scratchpad(self) -> str:
        """Format scratchpad for context."""
        import json

        return json.dumps(self._scratchpad, indent=2, default=str)[:1000]

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from LLM response."""
        import json

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {"reasoning": response, "action": "conclude", "conclusion": response}
