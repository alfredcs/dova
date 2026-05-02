"""
Synthesis Agent for DOVA.

Responsible for synthesizing research findings from multiple sources
into coherent, actionable insights.
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
class SynthesizedInsight:
    """A synthesized insight from research findings."""

    title: str
    summary: str
    confidence: float
    sources: list[str] = field(default_factory=list)
    related_topics: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class ResearchSynthesis:
    """Complete research synthesis."""

    executive_summary: str
    key_findings: list[SynthesizedInsight] = field(default_factory=list)
    knowledge_gaps: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    cross_references: dict[str, list[str]] = field(default_factory=dict)
    confidence_score: float = 0.0


class SynthesisAgent(BaseAgent):
    """
    Synthesis Agent for research result aggregation.

    Combines findings from multiple sources (ArXiv, GitHub, HuggingFace)
    into a coherent research synthesis.
    """

    @property
    def system_prompt(self) -> str:
        return """You are a Research Synthesis Agent specialized in combining and analyzing research findings.

Your synthesis capabilities:
1. Identify common themes across papers, code, and models
2. Detect contradictions or conflicting approaches
3. Find knowledge gaps and research opportunities
4. Generate actionable recommendations

When synthesizing:
- Cross-reference papers with their implementations
- Connect models with their training datasets and papers
- Identify the most influential and recent works
- Assess the maturity and adoption of approaches

Provide balanced, objective analysis with clear confidence levels."""

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute synthesis task."""
        start_time = time.time()

        try:
            task_type = task.type or "synthesize"
            query = task.params.get("query", "")
            dependency_results = task.context.get("dependency_results", {})

            self._logger.info(
                "synthesis_starting",
                query=query,
                source_count=len(dependency_results),
            )

            if task_type == "synthesize":
                synthesis = await self._synthesize_results(query, dependency_results)
                # Two-pass insight generation for deep analysis
                enable_two_pass = task.params.get("enable_two_pass", False)
                if enable_two_pass and synthesis.confidence_score > 0.4:
                    emergent = await self._generate_emergent_insights(query, synthesis)
                    synthesis.key_findings.extend(emergent)
                    self._logger.info("emergent_insights_added", count=len(emergent))
            elif task_type == "cross_reference":
                synthesis = await self._cross_reference(dependency_results)
            elif task_type == "gap_analysis":
                synthesis = await self._gap_analysis(query, dependency_results)
            else:
                return self._wrap_result(task, False, error=f"Unknown task type: {task_type}")

            execution_time = (time.time() - start_time) * 1000
            return self._wrap_result(
                task,
                True,
                data={
                    "executive_summary": synthesis.executive_summary,
                    "key_findings": [
                        {
                            "title": f.title,
                            "summary": f.summary,
                            "rationale": f.rationale,
                            "confidence": f.confidence,
                            "sources": f.sources,
                            "related_topics": f.related_topics,
                        }
                        for f in synthesis.key_findings
                    ],
                    "knowledge_gaps": synthesis.knowledge_gaps,
                    "recommendations": synthesis.recommendations,
                    "cross_references": synthesis.cross_references,
                    "confidence_score": synthesis.confidence_score,
                },
                execution_time_ms=execution_time,
            )

        except Exception as e:
            self._logger.exception("synthesis_error", error=str(e))
            return self._wrap_result(
                task,
                False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _synthesize_results(
        self,
        query: str,
        dependency_results: dict[str, Any],
    ) -> ResearchSynthesis:
        """Synthesize results from multiple research sources."""
        # Format results for synthesis
        formatted_results = self._format_dependency_results(dependency_results)

        synthesis_prompt = f"""Synthesize these research findings for the query: "{query}"

{formatted_results}

Provide a comprehensive synthesis including:

1. EXECUTIVE SUMMARY (2-3 paragraphs)
   - What is the current state of this research area?
   - What are the dominant approaches and their trade-offs?
   - What is the trajectory of the field (emerging vs mature)?

2. KEY FINDINGS (3-5 findings)
   For each finding:
   - title: concise title
   - summary: 3-5 sentence summary covering the core contribution, methodology or approach used, and significance
   - rationale: why this finding matters — explain the practical impact, how it advances the field, or why a practitioner should care
   - confidence: 0.0-1.0 based on evidence strength
   - sources: which sources support this (include URLs when available)
   - related_topics: connected research areas

3. KNOWLEDGE GAPS
   - What questions remain unanswered?
   - What areas lack implementation or validation?

4. RECOMMENDATIONS (3-5)
   - Actionable next steps with rationale for why each is worth pursuing
   - Promising directions to explore and what evidence supports them
   - When suggesting implementation approaches, use PDL (Program Description Language) style pseudo code — NOT Python or any specific language

5. CROSS-REFERENCES
   - papers_to_code: which papers have implementations
   - code_to_models: which repos produced models
   - models_to_papers: which models cite papers

Respond in JSON format with these sections."""

        response = await self.think(
            synthesis_prompt,
            task_type=TaskType.SUMMARIZATION,
            temperature=0.5,
            max_tokens=40000,
        )

        return self._parse_synthesis_response(response)

    async def _cross_reference(
        self,
        dependency_results: dict[str, Any],
    ) -> ResearchSynthesis:
        """Find cross-references between papers, code, and models."""
        formatted_results = self._format_dependency_results(dependency_results)

        cross_ref_prompt = f"""Analyze these research findings and identify cross-references:

{formatted_results}

Find connections between:
1. Papers that have code implementations
2. Repositories that implement published papers
3. Models that are based on specific papers
4. Datasets used across multiple papers/models

For each connection, note:
- The source items being connected
- The type of connection (implementation, citation, derivative)
- Confidence in the connection (certain/likely/possible)

Respond in JSON with:
{{
    "papers_to_code": [
        {{"paper": "<title>", "repo": "<repo>", "confidence": "<level>"}}
    ],
    "code_to_models": [...],
    "papers_to_models": [...],
    "shared_datasets": [
        {{"dataset": "<name>", "users": ["<paper1>", "<model1>"]}}
    ]
}}"""

        response = await self.think(
            cross_ref_prompt,
            task_type=TaskType.REASONING,
            temperature=0.3,
        )

        # Parse and build synthesis focused on cross-references
        import json

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            cross_refs = json.loads(response.strip())
        except json.JSONDecodeError:
            cross_refs = {}

        return ResearchSynthesis(
            executive_summary="Cross-reference analysis complete",
            cross_references=cross_refs,
            confidence_score=0.7,
        )

    async def _gap_analysis(
        self,
        query: str,
        dependency_results: dict[str, Any],
    ) -> ResearchSynthesis:
        """Identify knowledge gaps in the research area."""
        formatted_results = self._format_dependency_results(dependency_results)

        gap_prompt = f"""Analyze the knowledge gaps in this research area: "{query}"

Current findings:
{formatted_results}

Identify:
1. THEORETICAL GAPS
   - Concepts that are poorly understood
   - Missing theoretical frameworks

2. IMPLEMENTATION GAPS
   - Papers without code implementations
   - Methods that lack practical validation

3. EVALUATION GAPS
   - Missing benchmarks or datasets
   - Inconsistent evaluation methodologies

4. SCALABILITY GAPS
   - Methods untested at scale
   - Resource requirements unclear

5. APPLICATION GAPS
   - Real-world applications not explored
   - Domain-specific adaptations needed

For each gap, provide:
- description: what is missing
- impact: why it matters (high/medium/low)
- difficulty: how hard to address (high/medium/low)
- suggested_approach: how to address it

Respond in JSON format."""

        response = await self.think(
            gap_prompt,
            task_type=TaskType.REASONING,
            temperature=0.5,
        )

        import json

        knowledge_gaps = []
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            gaps_data = json.loads(response.strip())

            for category, gaps in gaps_data.items():
                if isinstance(gaps, list):
                    for gap in gaps:
                        if isinstance(gap, dict):
                            knowledge_gaps.append(
                                f"[{category}] {gap.get('description', str(gap))}"
                            )
                        else:
                            knowledge_gaps.append(f"[{category}] {gap}")
        except json.JSONDecodeError:
            knowledge_gaps = [response]

        return ResearchSynthesis(
            executive_summary=f"Gap analysis for: {query}",
            knowledge_gaps=knowledge_gaps,
            confidence_score=0.6,
        )

    def _format_dependency_results(self, dependency_results: dict[str, Any]) -> str:
        """Format dependency results for synthesis prompts."""
        sections = []

        for source, result in dependency_results.items():
            if result is None:
                continue

            # Handle AgentResult objects
            if hasattr(result, "success"):
                if not result.success:
                    sections.append(f"\n## {source}: Error - {result.error}")
                    continue
                data = result.data
            else:
                data = result

            if data is None:
                continue

            sections.append(f"\n## {source}:")

            # Handle ResearchFindings dataclass
            if hasattr(data, "papers"):
                if data.papers:
                    sections.append("Papers:")
                    for p in data.papers[:5]:
                        meta = getattr(p, "metadata", {}) or {}
                        authors = meta.get("authors", [])
                        author_str = f" | Authors: {', '.join(authors[:3])}" if authors else ""
                        categories = meta.get("categories", [])
                        cat_str = f" | Categories: {', '.join(categories[:3])}" if categories else ""
                        url = getattr(p, "url", "")
                        url_str = f" | URL: {url}" if url else ""
                        sections.append(
                            f"  - {p.title}{author_str}{cat_str}{url_str}\n"
                            f"    {p.description[:300]}"
                        )
                if data.repositories:
                    sections.append("Repositories:")
                    for r in data.repositories[:5]:
                        meta = getattr(r, "metadata", {}) or {}
                        stars = meta.get("stars", "")
                        lang = meta.get("language", "")
                        detail = f" ({stars} stars, {lang})" if stars else ""
                        url = getattr(r, "url", "")
                        url_str = f" | URL: {url}" if url else ""
                        sections.append(
                            f"  - {r.title}{detail}{url_str}\n"
                            f"    {r.description[:200]}"
                        )
                if data.models:
                    sections.append("Models:")
                    for m in data.models[:5]:
                        meta = getattr(m, "metadata", {}) or {}
                        downloads = meta.get("downloads", "")
                        pipeline = meta.get("pipeline_tag", "")
                        detail = f" ({downloads} downloads, {pipeline})" if downloads else ""
                        desc = getattr(m, "description", "")
                        desc_str = f"\n    {desc[:200]}" if desc else ""
                        sections.append(f"  - {m.title}{detail}{desc_str}")
            elif isinstance(data, dict):
                sections.append(str(data)[:1000])
            elif isinstance(data, str):
                sections.append(data[:1000])

        return "\n".join(sections) if sections else "No results available"

    async def _generate_emergent_insights(
        self,
        query: str,
        first_pass: ResearchSynthesis,
    ) -> list[SynthesizedInsight]:
        """Second pass: Extended thinking to find non-obvious insights."""
        insight_prompt = f"""Given research on: "{query}"

FIRST PASS SUMMARY:
{first_pass.executive_summary}

FINDINGS SO FAR:
{[f.title for f in first_pass.key_findings[:5]]}

TASK: Find 2-3 EMERGENT insights NOT in the first pass.
Look for:
1. Non-obvious connections between disparate findings
2. Contradictions that reveal deeper truths
3. Patterns suggesting emerging trends
4. Implications extending beyond the question

Return JSON:
{{
    "emergent_insights": [
        {{
            "title": "<insight>",
            "summary": "<2-3 sentences>",
            "novelty_score": <0.0-1.0>
        }}
    ]
}}"""

        response = await self.think(
            insight_prompt,
            task_type=TaskType.REASONING,
            temperature=0.7,
            max_tokens=24000,
        )

        return self._parse_emergent_insights(response)

    def _parse_emergent_insights(self, response: str) -> list[SynthesizedInsight]:
        """Parse emergent insights from LLM response."""
        import json
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            data = json.loads(response.strip())
            return [
                SynthesizedInsight(
                    title=i.get("title", ""),
                    summary=i.get("summary", ""),
                    confidence=i.get("novelty_score", 0.5),
                    sources=["emergent_analysis"],
                    related_topics=[],
                )
                for i in data.get("emergent_insights", [])
            ]
        except (json.JSONDecodeError, KeyError):
            return []

    def _parse_synthesis_response(self, response: str) -> ResearchSynthesis:
        """Parse LLM synthesis response."""
        import json

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            data = json.loads(response.strip())

            key_findings = []
            for finding in data.get("key_findings", data.get("KEY FINDINGS", [])):
                if isinstance(finding, dict):
                    key_findings.append(
                        SynthesizedInsight(
                            title=finding.get("title", ""),
                            summary=finding.get("summary", ""),
                            confidence=finding.get("confidence", 0.5),
                            sources=finding.get("sources", []),
                            related_topics=finding.get("related_topics", []),
                            rationale=finding.get("rationale", ""),
                        )
                    )

            return ResearchSynthesis(
                executive_summary=data.get(
                    "executive_summary",
                    data.get("EXECUTIVE SUMMARY", "Synthesis complete"),
                ),
                key_findings=key_findings,
                knowledge_gaps=data.get(
                    "knowledge_gaps", data.get("KNOWLEDGE GAPS", [])
                ),
                recommendations=data.get(
                    "recommendations", data.get("RECOMMENDATIONS", [])
                ),
                cross_references=data.get(
                    "cross_references", data.get("CROSS-REFERENCES", {})
                ),
                confidence_score=0.7,
            )

        except json.JSONDecodeError:
            self._logger.warning("synthesis_parse_error")
            return ResearchSynthesis(
                executive_summary=response[:1000],
                confidence_score=0.5,
            )
