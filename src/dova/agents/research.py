"""
Research Agent for DOVA.

Handles research queries across multiple sources:
- ArXiv papers
- GitHub repositories and code
- HuggingFace models and datasets
- Web search (via Tavily)
"""

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from dova.agents.base import AgentResult, AgentTask, BaseAgent
from dova.config.providers import LLMRouter, TaskType
from dova.services.sources import SourceRegistry, SourceFetcher
from dova.services.source_types import SourceType
from dova.utils.metrics import MetricsCollector

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """A single search result from any source."""

    source: str
    title: str
    url: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0


@dataclass
class ResearchFindings:
    """Aggregated research findings."""

    papers: list[SearchResult] = field(default_factory=list)
    repositories: list[SearchResult] = field(default_factory=list)
    models: list[SearchResult] = field(default_factory=list)
    datasets: list[SearchResult] = field(default_factory=list)
    web_results: list[SearchResult] = field(default_factory=list)


class ResearchAgent(BaseAgent):
    """
    Research Agent for multi-source research queries.

    Searches ArXiv, GitHub, HuggingFace, and web to gather
    comprehensive research on a topic.
    """

    def __init__(
        self,
        llm_router: Any,
        mcp_client: Any | None = None,
        metrics: MetricsCollector | None = None,
        memory_service: Any | None = None,
        source_registry: SourceRegistry | None = None,
    ):
        super().__init__(llm_router, mcp_client, metrics, memory_service=memory_service)
        self.source_registry = source_registry
        self.source_fetcher = SourceFetcher()

    @property
    def system_prompt(self) -> str:
        return """You are a Research Agent specialized in finding and analyzing academic papers, code repositories, ML models, and datasets.

Your capabilities:
1. Search ArXiv for relevant papers
2. Search GitHub for code implementations
3. Search HuggingFace for models and datasets
4. Synthesize findings across sources

When analyzing search results:
- Identify the most relevant and high-quality results
- Extract key findings, methods, and contributions
- Note connections between papers, code, and models
- Assess quality based on citations, stars, downloads, etc."""

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute research task."""
        start_time = time.time()

        try:
            source = task.params.get("source", "all")
            query = task.params.get("query", "")
            entities = task.params.get("entities", {})

            if not query:
                return self._wrap_result(task, False, error="No query provided")

            self._logger.info("research_starting", source=source, query=query)

            # Route to appropriate search method
            if source == "arxiv":
                results = await self._search_arxiv(query, entities)
            elif source == "github":
                results = await self._search_github(query, entities)
            elif source == "huggingface":
                results = await self._search_huggingface(query, entities)
            elif source == "all":
                results = await self._search_all(query, entities, user_id=task.user_id)
            else:
                return self._wrap_result(task, False, error=f"Unknown source: {source}")

            # Store query to memory
            if task.user_id and self.memory_service:
                await self.remember(
                    content={
                        "type": "search_query",
                        "query": query,
                        "source": source,
                        "results_count": self._count_results(results),
                    },
                    user_id=task.user_id,
                    session_id=task.id,
                    short_term=True,
                )

            execution_time = (time.time() - start_time) * 1000
            return self._wrap_result(
                task,
                True,
                data=results,
                execution_time_ms=execution_time,
                source=source,
                result_count=self._count_results(results),
            )

        except Exception as e:
            self._logger.exception("research_error", error=str(e))
            return self._wrap_result(
                task,
                False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _search_arxiv(
        self,
        query: str,
        entities: dict[str, Any],
    ) -> ResearchFindings:
        """Search ArXiv for papers."""
        findings = ResearchFindings()

        # Build optimized query using entities
        search_query = self._build_arxiv_query(query, entities)

        result = await self.search_arxiv(search_query, max_results=20)

        if result.success and result.data:
            papers = result.data if isinstance(result.data, list) else [result.data]
            for paper in papers:
                findings.papers.append(
                    SearchResult(
                        source="arxiv",
                        title=paper.get("title", ""),
                        url=paper.get("url", paper.get("id", "")),
                        description=paper.get("summary", paper.get("abstract", "")),
                        metadata={
                            "authors": paper.get("authors", []),
                            "categories": paper.get("categories", []),
                            "published": paper.get("published", ""),
                            "arxiv_id": paper.get("id", ""),
                        },
                    )
                )

        return findings

    async def _search_github(
        self,
        query: str,
        entities: dict[str, Any],
    ) -> ResearchFindings:
        """Search GitHub for repositories and code."""
        findings = ResearchFindings()

        # Search repositories
        search_query = self._build_github_query(query, entities)
        result = await self.search_github(search_query, search_type="repositories", per_page=20)

        if result.success and result.data:
            repos = result.data.get("items", result.data) if isinstance(result.data, dict) else result.data
            if isinstance(repos, list):
                for repo in repos:
                    findings.repositories.append(
                        SearchResult(
                            source="github",
                            title=repo.get("full_name", repo.get("name", "")),
                            url=repo.get("html_url", repo.get("url", "")),
                            description=repo.get("description", ""),
                            metadata={
                                "stars": repo.get("stargazers_count", 0),
                                "forks": repo.get("forks_count", 0),
                                "language": repo.get("language", ""),
                                "topics": repo.get("topics", []),
                                "updated_at": repo.get("updated_at", ""),
                            },
                            relevance_score=repo.get("stargazers_count", 0) / 1000,
                        )
                    )

        return findings

    async def _search_huggingface(
        self,
        query: str,
        entities: dict[str, Any],
    ) -> ResearchFindings:
        """Search HuggingFace for models and datasets."""
        findings = ResearchFindings()

        # Search models
        model_result = await self.search_huggingface(query, search_type="models", limit=15)
        if model_result.success and model_result.data:
            models = model_result.data if isinstance(model_result.data, list) else [model_result.data]
            for model in models:
                findings.models.append(
                    SearchResult(
                        source="huggingface",
                        title=model.get("id", model.get("modelId", "")),
                        url=f"https://huggingface.co/{model.get('id', model.get('modelId', ''))}",
                        description=model.get("description", ""),
                        metadata={
                            "downloads": model.get("downloads", 0),
                            "likes": model.get("likes", 0),
                            "tags": model.get("tags", []),
                            "library": model.get("library_name", ""),
                            "pipeline_tag": model.get("pipeline_tag", ""),
                        },
                        relevance_score=model.get("downloads", 0) / 100000,
                    )
                )

        # Search datasets
        dataset_result = await self.search_huggingface(query, search_type="datasets", limit=10)
        if dataset_result.success and dataset_result.data:
            datasets = dataset_result.data if isinstance(dataset_result.data, list) else [dataset_result.data]
            for dataset in datasets:
                findings.datasets.append(
                    SearchResult(
                        source="huggingface",
                        title=dataset.get("id", dataset.get("datasetId", "")),
                        url=f"https://huggingface.co/datasets/{dataset.get('id', dataset.get('datasetId', ''))}",
                        description=dataset.get("description", ""),
                        metadata={
                            "downloads": dataset.get("downloads", 0),
                            "likes": dataset.get("likes", 0),
                            "tags": dataset.get("tags", []),
                        },
                    )
                )

        return findings

    async def _search_custom_sources(
        self, query: str, user_id: str
    ) -> list[SearchResult]:
        """Search all enabled custom sources for a user."""
        if not self.source_registry:
            return []

        sources = await self.source_registry.get_sources(user_id, enabled_only=True)
        custom_sources = [s for s in sources if s.source_type != SourceType.BUILTIN]

        results = []
        for source in custom_sources:
            try:
                items = await self.source_fetcher.fetch(source, query)
                for item in items:
                    results.append(SearchResult(
                        source=source.id,
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        description=item.get("description", ""),
                        metadata={"source_name": source.name, "source_type": source.source_type.value},
                        relevance_score=source.quality.quality_score,
                    ))
            except Exception as e:
                self._logger.warning("custom_source_error", source=source.id, error=str(e))

        return results

    async def _search_all(
        self,
        query: str,
        entities: dict[str, Any],
        user_id: str | None = None,
    ) -> ResearchFindings:
        """Search all sources in parallel."""
        import asyncio

        tasks = [
            self._search_arxiv(query, entities),
            self._search_github(query, entities),
            self._search_huggingface(query, entities),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge findings
        findings = ResearchFindings()

        for result in results:
            if isinstance(result, Exception):
                self._logger.warning("search_partial_failure", error=str(result))
                continue
            if isinstance(result, ResearchFindings):
                findings.papers.extend(result.papers)
                findings.repositories.extend(result.repositories)
                findings.models.extend(result.models)
                findings.datasets.extend(result.datasets)
                findings.web_results.extend(result.web_results)

        # Add custom sources if user_id provided
        if user_id and self.source_registry:
            try:
                custom_results = await self._search_custom_sources(query, user_id)
                findings.web_results.extend(custom_results)
                # Sort web_results by quality score
                findings.web_results.sort(key=lambda r: r.relevance_score, reverse=True)
            except Exception as e:
                self._logger.warning("custom_sources_error", error=str(e))

        return findings

    def _build_arxiv_query(self, query: str, entities: dict[str, Any]) -> str:
        """Build optimized ArXiv query from entities."""
        # Start with base query
        parts = [query]

        # Add author filter if specified
        authors = entities.get("authors", [])
        if authors:
            parts.append(f"au:{authors[0]}")

        # Add category filter for known topics
        topics = entities.get("topics", [])
        topic_to_category = {
            "machine learning": "cs.LG",
            "deep learning": "cs.LG",
            "nlp": "cs.CL",
            "natural language": "cs.CL",
            "computer vision": "cs.CV",
            "reinforcement learning": "cs.LG",
            "transformers": "cs.CL",
            "llm": "cs.CL",
        }
        for topic in topics:
            topic_lower = topic.lower()
            for key, category in topic_to_category.items():
                if key in topic_lower:
                    parts.append(f"cat:{category}")
                    break

        return " AND ".join(parts) if len(parts) > 1 else query

    def _build_github_query(self, query: str, entities: dict[str, Any]) -> str:
        """Build optimized GitHub query from entities."""
        # Extract keywords from topics if available, otherwise clean the query
        topics = entities.get("topics", [])
        if topics:
            # Use extracted topics as the main search terms
            search_terms = " ".join(topics[:3])  # Limit to top 3 topics
        else:
            # Clean natural language query to extract keywords
            search_terms = self._extract_search_keywords(query)

        parts = [search_terms]

        # Add language filter
        technologies = entities.get("technologies", [])
        lang_map = {"python": "python", "javascript": "javascript", "typescript": "typescript", "rust": "rust"}
        for tech in technologies:
            tech_lower = tech.lower()
            if tech_lower in lang_map:
                parts.append(f"language:{lang_map[tech_lower]}")
                break

        # Add date filter if time_range is specified
        time_range = entities.get("time_range")
        if time_range:
            # Format: pushed:>2025-01-01
            parts.append(f"pushed:>{time_range}")

        # Add minimum stars for quality (but lower threshold for broader results)
        parts.append("stars:>5")

        return " ".join(parts)

    def _extract_search_keywords(self, query: str) -> str:
        """Extract meaningful keywords from natural language query."""
        import re

        # Remove common stop words and phrases
        stop_phrases = [
            r'\bshow me\b', r'\bfind\b', r'\bsearch for\b', r'\blooking for\b',
            r'\bthe\b', r'\ba\b', r'\ban\b', r'\bwhat is\b', r'\bwhat are\b',
            r'\bhow to\b', r'\bpublish\b', r'\bpublished\b', r'\brepo\b',
            r'\brepository\b', r'\brepositories\b', r'\bsince\b', r'\bfrom\b',
            r'\bon\b', r'\babout\b', r'\brelated to\b', r'\bme\b',
        ]

        cleaned = query.lower()
        for phrase in stop_phrases:
            cleaned = re.sub(phrase, ' ', cleaned, flags=re.IGNORECASE)

        # Remove date patterns like "Oct 2025", "October 2025", "2025-01"
        cleaned = re.sub(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4}\b', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\b\d{4}[-/]\d{1,2}([-/]\d{1,2})?\b', '', cleaned)
        cleaned = re.sub(r'\b\d{4}\b', '', cleaned)

        # Clean up extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned if cleaned else query

    def _count_results(self, findings: ResearchFindings) -> int:
        """Count total results in findings."""
        return (
            len(findings.papers)
            + len(findings.repositories)
            + len(findings.models)
            + len(findings.datasets)
            + len(findings.web_results)
        )

    async def analyze_findings(
        self,
        findings: ResearchFindings,
        query: str,
    ) -> dict[str, Any]:
        """Use LLM to analyze and summarize research findings."""
        analysis_prompt = f"""Analyze these research findings for the query: "{query}"

Papers found: {len(findings.papers)}
{self._format_papers(findings.papers[:5])}

Repositories found: {len(findings.repositories)}
{self._format_repos(findings.repositories[:5])}

Models found: {len(findings.models)}
{self._format_models(findings.models[:5])}

Provide analysis including:
1. Key themes and trends
2. Most influential/relevant papers
3. Best implementations
4. Recommended models
5. Knowledge gaps or areas for further research"""

        analysis = await self.think(
            analysis_prompt,
            task_type=TaskType.SUMMARIZATION,
            temperature=0.5,
        )

        return {
            "analysis": analysis,
            "paper_count": len(findings.papers),
            "repo_count": len(findings.repositories),
            "model_count": len(findings.models),
            "dataset_count": len(findings.datasets),
        }

    def _format_papers(self, papers: list[SearchResult]) -> str:
        """Format papers for prompt."""
        if not papers:
            return "None found"
        return "\n".join(
            f"- {p.title}: {p.description[:200]}..." for p in papers
        )

    def _format_repos(self, repos: list[SearchResult]) -> str:
        """Format repositories for prompt."""
        if not repos:
            return "None found"
        return "\n".join(
            f"- {r.title} ({r.metadata.get('stars', 0)} stars): {r.description[:100]}..."
            for r in repos
        )

    def _format_models(self, models: list[SearchResult]) -> str:
        """Format models for prompt."""
        if not models:
            return "None found"
        return "\n".join(
            f"- {m.title} ({m.metadata.get('downloads', 0)} downloads)"
            for m in models
        )
