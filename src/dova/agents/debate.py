"""
Debate Agents for DOVA.

Implements Bull vs Bear debate mechanism for balanced analysis:
- BullAgent: Advocates for the positive aspects
- BearAgent: Provides critical counterarguments
- DebateAgent: Orchestrates debate and synthesizes balanced conclusions
"""

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from dova.agents.base import AgentResult, AgentTask, BaseAgent
from dova.config.providers import LLMRouter, TaskType
from dova.utils.metrics import MetricsCollector

logger = structlog.get_logger(__name__)


@dataclass
class DebateArgument:
    """A single argument in the debate."""

    position: str  # "bull" or "bear"
    round: int
    argument: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class DebateConclusion:
    """Synthesized conclusion from debate."""

    summary: str
    bull_strengths: list[str] = field(default_factory=list)
    bear_concerns: list[str] = field(default_factory=list)
    balanced_assessment: str = ""
    recommendation: str = ""
    confidence_score: float = 0.0


class BullAgent(BaseAgent):
    """
    Bull Agent - Advocates for positive aspects.

    Finds strengths, potential, and optimistic perspectives
    on research topics, technologies, or solutions.
    """

    @property
    def system_prompt(self) -> str:
        return """You are a Bull Agent in a structured debate. Your role is to advocate for the positive aspects.

Your approach:
1. Identify strengths and advantages
2. Highlight potential and opportunities
3. Find evidence supporting adoption
4. Counter bear arguments constructively

Be persuasive but honest - acknowledge limitations while emphasizing benefits.
Support arguments with specific evidence from research findings."""

    async def execute(self, task: AgentTask) -> AgentResult:
        """Generate bull (positive) arguments."""
        start_time = time.time()

        try:
            topic = task.params.get("topic", "")
            context = task.params.get("context", {})
            bear_arguments = task.params.get("bear_arguments", [])
            round_num = task.params.get("round", 1)

            if not topic:
                return self._wrap_result(task, False, error="No topic provided")

            argument = await self._generate_argument(
                topic, context, bear_arguments, round_num
            )

            execution_time = (time.time() - start_time) * 1000
            return self._wrap_result(
                task,
                True,
                data={
                    "position": "bull",
                    "round": round_num,
                    "argument": argument.argument,
                    "evidence": argument.evidence,
                    "confidence": argument.confidence,
                },
                execution_time_ms=execution_time,
            )

        except Exception as e:
            self._logger.exception("bull_agent_error", error=str(e))
            return self._wrap_result(
                task,
                False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _generate_argument(
        self,
        topic: str,
        context: dict[str, Any],
        bear_arguments: list[str],
        round_num: int,
    ) -> DebateArgument:
        """Generate bull arguments for the topic."""
        if round_num == 1:
            prompt = f"""Present the BULL case for: "{topic}"

Context and evidence:
{self._format_context(context)}

Provide:
1. Main argument (2-3 sentences)
2. Supporting evidence (3-5 points)
3. Confidence level (0.0-1.0)

Focus on:
- Technical strengths and innovations
- Practical benefits and applications
- Market adoption and community support
- Future potential and roadmap

Respond in JSON:
{{
    "argument": "<main argument>",
    "evidence": ["<point1>", "<point2>", ...],
    "confidence": <0.0-1.0>
}}"""
        else:
            prompt = f"""Continue the BULL case for: "{topic}"

Previous bear arguments to address:
{bear_arguments}

Context:
{self._format_context(context)}

Counter the bear arguments while strengthening your position.

Respond in JSON:
{{
    "argument": "<counter argument>",
    "evidence": ["<point1>", "<point2>", ...],
    "confidence": <0.0-1.0>
}}"""

        response = await self.think(prompt, task_type=TaskType.REASONING, temperature=0.6)

        return self._parse_argument_response(response, "bull", round_num)

    def _format_context(self, context: dict[str, Any]) -> str:
        """Format context for prompts."""
        if not context:
            return "No additional context provided."
        return "\n".join(f"- {k}: {v}" for k, v in context.items())

    def _parse_argument_response(
        self, response: str, position: str, round_num: int
    ) -> DebateArgument:
        """Parse argument response from LLM."""
        import json

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            data = json.loads(response.strip())
            return DebateArgument(
                position=position,
                round=round_num,
                argument=data.get("argument", ""),
                evidence=data.get("evidence", []),
                confidence=data.get("confidence", 0.5),
            )
        except json.JSONDecodeError:
            return DebateArgument(
                position=position,
                round=round_num,
                argument=response,
                confidence=0.5,
            )


class BearAgent(BaseAgent):
    """
    Bear Agent - Provides critical analysis.

    Identifies weaknesses, risks, and concerns about
    research topics, technologies, or solutions.
    """

    @property
    def system_prompt(self) -> str:
        return """You are a Bear Agent in a structured debate. Your role is to provide critical analysis.

Your approach:
1. Identify weaknesses and limitations
2. Highlight risks and concerns
3. Question assumptions and hype
4. Counter bull arguments with evidence

Be rigorous but fair - acknowledge strengths while emphasizing concerns.
Support arguments with specific evidence and logical reasoning."""

    async def execute(self, task: AgentTask) -> AgentResult:
        """Generate bear (critical) arguments."""
        start_time = time.time()

        try:
            topic = task.params.get("topic", "")
            context = task.params.get("context", {})
            bull_arguments = task.params.get("bull_arguments", [])
            round_num = task.params.get("round", 1)

            if not topic:
                return self._wrap_result(task, False, error="No topic provided")

            argument = await self._generate_argument(
                topic, context, bull_arguments, round_num
            )

            execution_time = (time.time() - start_time) * 1000
            return self._wrap_result(
                task,
                True,
                data={
                    "position": "bear",
                    "round": round_num,
                    "argument": argument.argument,
                    "evidence": argument.evidence,
                    "confidence": argument.confidence,
                },
                execution_time_ms=execution_time,
            )

        except Exception as e:
            self._logger.exception("bear_agent_error", error=str(e))
            return self._wrap_result(
                task,
                False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _generate_argument(
        self,
        topic: str,
        context: dict[str, Any],
        bull_arguments: list[str],
        round_num: int,
    ) -> DebateArgument:
        """Generate bear arguments for the topic."""
        if round_num == 1:
            prompt = f"""Present the BEAR case for: "{topic}"

Context and evidence:
{self._format_context(context)}

Provide:
1. Main argument (2-3 sentences)
2. Supporting evidence (3-5 points)
3. Confidence level (0.0-1.0)

Focus on:
- Technical limitations and challenges
- Practical concerns and implementation difficulties
- Competition and alternatives
- Risks and potential downsides

Respond in JSON:
{{
    "argument": "<main argument>",
    "evidence": ["<point1>", "<point2>", ...],
    "confidence": <0.0-1.0>
}}"""
        else:
            prompt = f"""Continue the BEAR case for: "{topic}"

Previous bull arguments to address:
{bull_arguments}

Context:
{self._format_context(context)}

Counter the bull arguments while strengthening your position.

Respond in JSON:
{{
    "argument": "<counter argument>",
    "evidence": ["<point1>", "<point2>", ...],
    "confidence": <0.0-1.0>
}}"""

        response = await self.think(prompt, task_type=TaskType.REASONING, temperature=0.6)

        return self._parse_argument_response(response, "bear", round_num)

    def _format_context(self, context: dict[str, Any]) -> str:
        """Format context for prompts."""
        if not context:
            return "No additional context provided."
        return "\n".join(f"- {k}: {v}" for k, v in context.items())

    def _parse_argument_response(
        self, response: str, position: str, round_num: int
    ) -> DebateArgument:
        """Parse argument response from LLM."""
        import json

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            data = json.loads(response.strip())
            return DebateArgument(
                position=position,
                round=round_num,
                argument=data.get("argument", ""),
                evidence=data.get("evidence", []),
                confidence=data.get("confidence", 0.5),
            )
        except json.JSONDecodeError:
            return DebateArgument(
                position=position,
                round=round_num,
                argument=response,
                confidence=0.5,
            )


class DebateAgent(BaseAgent):
    """
    Debate Agent - Orchestrates Bull vs Bear debates.

    Coordinates multiple rounds of debate between Bull and Bear agents,
    then synthesizes a balanced conclusion.
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        mcp_client: Any | None = None,
        metrics: MetricsCollector | None = None,
        num_rounds: int = 2,
    ):
        super().__init__(llm_router, mcp_client, metrics)
        self.num_rounds = num_rounds
        self.bull_agent = BullAgent(llm_router, mcp_client, metrics)
        self.bear_agent = BearAgent(llm_router, mcp_client, metrics)

    @property
    def system_prompt(self) -> str:
        return """You are a Debate Moderator responsible for synthesizing balanced conclusions from Bull vs Bear debates.

Your role:
1. Evaluate arguments from both sides objectively
2. Identify the strongest points from each position
3. Synthesize a balanced, nuanced conclusion
4. Provide clear recommendations based on the evidence

Be impartial and thorough in your assessment."""

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute debate and synthesize conclusion."""
        start_time = time.time()

        try:
            topic = task.params.get("topic", "")
            context = task.params.get("context", {})

            if not topic:
                return self._wrap_result(task, False, error="No topic provided")

            self._logger.info("debate_starting", topic=topic, rounds=self.num_rounds)

            # Run debate rounds
            debate_history = await self._run_debate(topic, context)

            # Synthesize conclusion
            conclusion = await self._synthesize_conclusion(topic, debate_history)

            execution_time = (time.time() - start_time) * 1000
            return self._wrap_result(
                task,
                True,
                data={
                    "summary": conclusion.summary,
                    "bull_strengths": conclusion.bull_strengths,
                    "bear_concerns": conclusion.bear_concerns,
                    "balanced_assessment": conclusion.balanced_assessment,
                    "recommendation": conclusion.recommendation,
                    "confidence_score": conclusion.confidence_score,
                    "debate_history": [
                        {
                            "position": arg.position,
                            "round": arg.round,
                            "argument": arg.argument,
                        }
                        for arg in debate_history
                    ],
                },
                execution_time_ms=execution_time,
            )

        except Exception as e:
            self._logger.exception("debate_error", error=str(e))
            return self._wrap_result(
                task,
                False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _run_debate(
        self,
        topic: str,
        context: dict[str, Any],
    ) -> list[DebateArgument]:
        """Run multiple rounds of debate."""
        history: list[DebateArgument] = []
        bull_arguments: list[str] = []
        bear_arguments: list[str] = []

        for round_num in range(1, self.num_rounds + 1):
            self._logger.debug("debate_round", round=round_num)

            # Bull argues first
            bull_task = AgentTask(
                type="argue",
                params={
                    "topic": topic,
                    "context": context,
                    "bear_arguments": bear_arguments,
                    "round": round_num,
                },
            )
            bull_result = await self.bull_agent.execute(bull_task)
            if bull_result.success:
                bull_arg = DebateArgument(
                    position="bull",
                    round=round_num,
                    argument=bull_result.data.get("argument", ""),
                    evidence=bull_result.data.get("evidence", []),
                    confidence=bull_result.data.get("confidence", 0.5),
                )
                history.append(bull_arg)
                bull_arguments.append(bull_arg.argument)

            # Bear responds
            bear_task = AgentTask(
                type="argue",
                params={
                    "topic": topic,
                    "context": context,
                    "bull_arguments": bull_arguments,
                    "round": round_num,
                },
            )
            bear_result = await self.bear_agent.execute(bear_task)
            if bear_result.success:
                bear_arg = DebateArgument(
                    position="bear",
                    round=round_num,
                    argument=bear_result.data.get("argument", ""),
                    evidence=bear_result.data.get("evidence", []),
                    confidence=bear_result.data.get("confidence", 0.5),
                )
                history.append(bear_arg)
                bear_arguments.append(bear_arg.argument)

        return history

    async def _synthesize_conclusion(
        self,
        topic: str,
        debate_history: list[DebateArgument],
    ) -> DebateConclusion:
        """Synthesize balanced conclusion from debate."""
        # Format debate history
        debate_text = "\n\n".join(
            f"[{arg.position.upper()} - Round {arg.round}]\n{arg.argument}"
            for arg in debate_history
        )

        synthesis_prompt = f"""Synthesize a balanced conclusion from this Bull vs Bear debate:

Topic: "{topic}"

DEBATE TRANSCRIPT:
{debate_text}

Provide:
1. summary: 2-3 sentence overall summary
2. bull_strengths: Top 3 valid points from the bull side
3. bear_concerns: Top 3 valid concerns from the bear side
4. balanced_assessment: Nuanced 2-3 paragraph assessment weighing both sides
5. recommendation: Clear actionable recommendation
6. confidence_score: 0.0-1.0 confidence in your assessment

Be objective and acknowledge the merit in both positions.

Respond in JSON format."""

        response = await self.think(
            synthesis_prompt,
            task_type=TaskType.SUMMARIZATION,
            temperature=0.4,
        )

        return self._parse_conclusion(response)

    def _parse_conclusion(self, response: str) -> DebateConclusion:
        """Parse synthesis response into conclusion."""
        import json

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            data = json.loads(response.strip())
            return DebateConclusion(
                summary=data.get("summary", ""),
                bull_strengths=data.get("bull_strengths", []),
                bear_concerns=data.get("bear_concerns", []),
                balanced_assessment=data.get("balanced_assessment", ""),
                recommendation=data.get("recommendation", ""),
                confidence_score=data.get("confidence_score", 0.5),
            )
        except json.JSONDecodeError:
            return DebateConclusion(
                summary=response[:500],
                balanced_assessment=response,
                confidence_score=0.5,
            )
