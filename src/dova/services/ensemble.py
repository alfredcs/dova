"""
Ensemble Reasoning Service for DOVA.

Multiple agents tackle the same problem, results are aggregated.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class AggregationMethod(Enum):
    SYNTHESIS = "synthesis"  # LLM combines insights
    VOTE = "vote"  # Weighted voting
    BEST_OF = "best_of"  # Select highest confidence
    UNION = "union"  # Combine all unique insights


@dataclass
class AgentAnswer:
    """An individual agent's answer."""

    agent_name: str
    answer: str
    confidence: float
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleResult:
    """Result from ensemble reasoning."""

    synthesized_answer: str
    individual_answers: list[AgentAnswer]
    agreement_score: float  # How much agents agreed
    confidence: float
    dissenting_views: list[str] = field(default_factory=list)
    method_used: AggregationMethod = AggregationMethod.SYNTHESIS


class EnsembleReasoning:
    """Orchestrate ensemble of agents for robust reasoning."""

    def __init__(self, llm_func: Any = None):
        """
        Args:
            llm_func: Async function for LLM calls (for synthesis)
        """
        self.llm_func = llm_func
        self._logger = logger.bind(service="ensemble")

    async def reason(
        self,
        problem: str,
        agents: list[Any],
        method: AggregationMethod = AggregationMethod.SYNTHESIS,
        context: dict[str, Any] | None = None,
    ) -> EnsembleResult:
        """
        Have multiple agents reason about the same problem.

        Args:
            problem: The problem to solve
            agents: List of agent instances
            method: How to aggregate results
            context: Shared context for all agents

        Returns:
            EnsembleResult with synthesized answer
        """
        self._logger.info(
            "ensemble_start", problem=problem[:100], agent_count=len(agents)
        )

        # Collect answers from all agents in parallel
        tasks = [
            self._get_agent_answer(agent, problem, context) for agent in agents
        ]
        answers = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_answers = [a for a in answers if isinstance(a, AgentAnswer)]

        if not valid_answers:
            return EnsembleResult(
                synthesized_answer="No valid answers from ensemble",
                individual_answers=[],
                agreement_score=0.0,
                confidence=0.0,
                method_used=method,
            )

        # Aggregate based on method
        if method == AggregationMethod.BEST_OF:
            result = self._aggregate_best_of(valid_answers)
        elif method == AggregationMethod.VOTE:
            result = self._aggregate_vote(valid_answers)
        elif method == AggregationMethod.UNION:
            result = self._aggregate_union(valid_answers)
        else:  # SYNTHESIS
            result = await self._aggregate_synthesis(valid_answers, problem)

        result.method_used = method
        return result

    async def _get_agent_answer(
        self,
        agent: Any,
        problem: str,
        context: dict[str, Any] | None,
    ) -> AgentAnswer:
        """Get an answer from a single agent."""
        try:
            # Use reasoning if available, otherwise direct think
            if hasattr(agent, "reason"):
                trace = await agent.reason(problem, context=context)
                return AgentAnswer(
                    agent_name=agent.name,
                    answer=trace.refined_answer or trace.final_answer,
                    confidence=trace.confidence,
                    reasoning=trace.self_critique or "",
                )
            elif hasattr(agent, "think"):
                answer = await agent.think(f"Answer this: {problem}")
                return AgentAnswer(
                    agent_name=getattr(agent, "name", "unknown"),
                    answer=answer,
                    confidence=0.5,
                )
            else:
                raise ValueError(f"Agent {agent} has no think or reason method")
        except Exception as e:
            self._logger.warning(
                "agent_answer_failed", agent=str(agent), error=str(e)
            )
            raise

    def _aggregate_best_of(self, answers: list[AgentAnswer]) -> EnsembleResult:
        """Select highest confidence answer."""
        best = max(answers, key=lambda a: a.confidence)
        others = [a for a in answers if a != best]

        return EnsembleResult(
            synthesized_answer=best.answer,
            individual_answers=answers,
            agreement_score=self._calculate_agreement(answers),
            confidence=best.confidence,
            dissenting_views=[a.answer for a in others if a.confidence > 0.3],
        )

    def _aggregate_vote(self, answers: list[AgentAnswer]) -> EnsembleResult:
        """Weighted voting on answers."""
        # Group similar answers (simplified - in production use semantic similarity)
        total_weight = sum(a.confidence for a in answers)
        if total_weight == 0:
            return self._aggregate_best_of(answers)

        # Weighted average confidence
        avg_confidence = total_weight / len(answers)

        # Use highest confidence answer as representative
        best = max(answers, key=lambda a: a.confidence)

        return EnsembleResult(
            synthesized_answer=best.answer,
            individual_answers=answers,
            agreement_score=self._calculate_agreement(answers),
            confidence=avg_confidence,
        )

    def _aggregate_union(self, answers: list[AgentAnswer]) -> EnsembleResult:
        """Combine all unique insights."""
        combined = "\n\n".join(
            [
                f"[{a.agent_name}] (confidence: {a.confidence:.2f}): {a.answer}"
                for a in answers
            ]
        )

        return EnsembleResult(
            synthesized_answer=combined,
            individual_answers=answers,
            agreement_score=self._calculate_agreement(answers),
            confidence=sum(a.confidence for a in answers) / len(answers),
        )

    async def _aggregate_synthesis(
        self,
        answers: list[AgentAnswer],
        problem: str,
    ) -> EnsembleResult:
        """Use LLM to synthesize best insights."""
        if not self.llm_func:
            return self._aggregate_best_of(answers)

        prompt = f"""Synthesize these different perspectives on the problem into a single best answer.

Problem: {problem}

Perspectives:
{self._format_answers(answers)}

Create a synthesized answer that:
1. Incorporates the strongest points from each perspective
2. Resolves any contradictions
3. Notes areas of disagreement

Respond in JSON:
{{
    "synthesized_answer": "<your combined answer>",
    "incorporated_from": ["<agent1>", "<agent2>"],
    "disagreements": ["<point of disagreement>"],
    "confidence": <0.0-1.0>
}}"""

        response = await self.llm_func(prompt)
        parsed = self._parse_json(response)

        return EnsembleResult(
            synthesized_answer=parsed.get("synthesized_answer", answers[0].answer),
            individual_answers=answers,
            agreement_score=self._calculate_agreement(answers),
            confidence=parsed.get("confidence", 0.5),
            dissenting_views=parsed.get("disagreements", []),
        )

    def _calculate_agreement(self, answers: list[AgentAnswer]) -> float:
        """Calculate how much agents agreed (simplified)."""
        if len(answers) < 2:
            return 1.0

        # Simple heuristic: average confidence variance
        confidences = [a.confidence for a in answers]
        mean = sum(confidences) / len(confidences)
        variance = sum((c - mean) ** 2 for c in confidences) / len(confidences)

        # Lower variance = higher agreement
        return max(0.0, 1.0 - variance)

    def _format_answers(self, answers: list[AgentAnswer]) -> str:
        """Format answers for synthesis prompt."""
        return "\n\n".join(
            [
                f"[{a.agent_name}] (confidence: {a.confidence:.2f}):\n{a.answer}\nReasoning: {a.reasoning}"
                for a in answers
            ]
        )

    def _parse_json(self, response: str) -> dict[str, Any]:
        """Parse JSON from response."""
        import json

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {"synthesized_answer": response}
