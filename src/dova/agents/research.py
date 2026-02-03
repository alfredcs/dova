"""
Research Agent for DOVA.

Handles research queries across multiple sources:
- ArXiv papers
- GitHub repositories and code
- HuggingFace models and datasets
- Web search (multi-provider: Brave, Perplexity, Tavily, DuckDuckGo)
"""

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from dova.agents.base import AgentResult, AgentTask, BaseAgent
from dova.config.providers import LLMRouter, TaskType
from dova.services.sources import SourceRegistry, SourceFetcher
from dova.services.source_types import SourceType
from dova.services.web_search import WebSearchService, create_web_search_service
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
        tavily_api_key: str | None = None,
        web_search_service: WebSearchService | None = None,
        enhanced_memory_service: Any | None = None,
    ):
        super().__init__(llm_router, mcp_client, metrics, memory_service=memory_service)
        self.source_registry = source_registry
        self.source_fetcher = SourceFetcher()
        # Legacy Tavily support (deprecated - use web_search_service)
        self.tavily_api_key = tavily_api_key
        self._tavily_client = None
        # New multi-provider web search service
        self._web_search_service = web_search_service
        # Enhanced memory with short-term/long-term support
        self.enhanced_memory_service = enhanced_memory_service

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

    def _detect_query_intent(self, query: str) -> dict[str, Any]:
        """
        Detect user intent from the query to prioritize appropriate sources.

        Returns:
            dict with:
            - primary_source: The main source user is asking about (github, arxiv, huggingface, web, or None)
            - query_type: Type of query (technical, biographical, factual, general)
            - recommended_sources: List of sources to search based on query type
            - search_query: Cleaned query for searching
        """
        query_lower = query.lower()

        # === STEP 1: Classify Query Type ===
        # Technical/Research indicators (needs ArXiv, GitHub, HuggingFace)
        technical_keywords = [
            "algorithm", "architecture", "model", "neural network", "transformer",
            "machine learning", "deep learning", "ai", "artificial intelligence",
            "implementation", "framework", "library", "code", "api", "benchmark",
            "training", "inference", "llm", "gpt", "bert", "diffusion", "attention",
            "paper", "research", "method", "technique", "approach", "sota",
            "state of the art", "latest", "recent advances", "survey"
        ]

        # Biographical/Person indicators (needs web search primarily)
        biographical_keywords = [
            "who is", "who was", "biography", "born", "died", "age", "family",
            "married", "wife", "husband", "children", "parents", "education",
            "college", "university", "degree", "school", "background", "history",
            "career", "life", "early life", "personal", "net worth", "salary"
        ]

        # Factual/General knowledge indicators (needs web search)
        factual_keywords = [
            "what is", "what are", "when did", "where is", "how many", "how much",
            "explain", "define", "meaning", "history of", "origin of", "why",
            "difference between", "compare", "vs", "versus"
        ]

        # Person name detection (common patterns)
        person_indicators = [
            "'s ", "elon musk", "bill gates", "jeff bezos", "mark zuckerberg",
            "sam altman", "sundar pichai", "satya nadella", "jensen huang"
        ]

        # Count matches
        technical_score = sum(1 for kw in technical_keywords if kw in query_lower)
        biographical_score = sum(1 for kw in biographical_keywords if kw in query_lower)
        factual_score = sum(1 for kw in factual_keywords if kw in query_lower)
        person_score = sum(1 for kw in person_indicators if kw in query_lower)

        # Boost biographical if person name detected
        if person_score > 0:
            biographical_score += person_score * 2

        # Determine query type
        type_scores = {
            "technical": technical_score,
            "biographical": biographical_score,
            "factual": factual_score,
        }
        max_type_score = max(type_scores.values())
        if max_type_score == 0:
            query_type = "general"
        else:
            query_type = max(type_scores, key=type_scores.get)

        # === STEP 2: Determine Recommended Sources ===
        if query_type == "technical":
            recommended_sources = ["arxiv", "github", "huggingface", "web"]
        elif query_type == "biographical":
            recommended_sources = ["web"]  # Only web for biographical
        elif query_type == "factual":
            recommended_sources = ["web", "arxiv"]  # Web first, then arxiv for academic facts
        else:
            recommended_sources = ["web", "github", "arxiv", "huggingface"]

        # === STEP 3: Detect Primary Source (explicit mentions) ===
        github_keywords = [
            "github", "repo", "repository", "repositories", "starred", "stars",
            "fork", "forks", "code implementation", "source code", "open source",
            "most starred", "top repo", "popular repo"
        ]
        arxiv_keywords = [
            "arxiv", "paper", "papers", "research paper", "publication",
            "academic", "journal", "conference"
        ]
        hf_keywords = [
            "huggingface", "hugging face", "hf model", "pretrained model",
            "model hub", "transformers library", "checkpoint"
        ]
        web_keywords = [
            "website", "blog", "tutorial", "guide", "documentation"
        ]

        github_score = sum(1 for kw in github_keywords if kw in query_lower)
        arxiv_score = sum(1 for kw in arxiv_keywords if kw in query_lower)
        hf_score = sum(1 for kw in hf_keywords if kw in query_lower)
        web_score = sum(1 for kw in web_keywords if kw in query_lower)

        scores = {
            "github": github_score,
            "arxiv": arxiv_score,
            "huggingface": hf_score,
            "web": web_score,
        }
        max_score = max(scores.values())
        primary_source = None
        if max_score > 0:
            primary_source = max(scores, key=scores.get)

        # Override recommended sources if explicit source mentioned
        if primary_source:
            if primary_source not in recommended_sources:
                recommended_sources.insert(0, primary_source)

        # === STEP 4: Clean search query ===
        source_platform_words = [
            "github", "arxiv", "huggingface", "hugging face", "hf model",
            "on github", "from github", "in github",
            "on arxiv", "from arxiv",
            "on huggingface", "from huggingface",
        ]
        search_query = query_lower
        for kw in source_platform_words:
            search_query = search_query.replace(kw, " ")
        search_query = " ".join(search_query.split())

        self._logger.info(
            "intent_detected",
            original_query=query,
            query_type=query_type,
            primary_source=primary_source,
            recommended_sources=recommended_sources,
            type_scores=type_scores,
            search_query=search_query,
        )

        return {
            "primary_source": primary_source,
            "query_type": query_type,
            "recommended_sources": recommended_sources,
            "scores": scores,
            "search_query": search_query.strip() or query,
        }

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute research task."""
        start_time = time.time()

        try:
            source = task.params.get("source", "all")
            query = task.params.get("query", "")
            entities = task.params.get("entities", {})

            if not query:
                return self._wrap_result(task, False, error="No query provided")

            # Detect user intent to prioritize sources
            intent = self._detect_query_intent(query)
            entities["intent"] = intent

            self._logger.info(
                "research_starting",
                source=source,
                query=query,
                detected_intent=intent.get("primary_source"),
            )

            # Route to appropriate search method
            if source == "arxiv":
                results = await self._search_arxiv(query, entities)
            elif source == "github":
                results = await self._search_github(query, entities)
            elif source == "huggingface":
                results = await self._search_huggingface(query, entities)
            elif source == "web":
                results = await self._search_web(query, entities)
            elif source == "all":
                results = await self._search_all(query, entities, user_id=task.user_id)
            else:
                return self._wrap_result(task, False, error=f"Unknown source: {source}")

            # Store query to memory (enhanced or legacy)
            if task.user_id:
                memory_content = {
                    "type": "search_query",
                    "query": query,
                    "source": source,
                    "results_count": self._count_results(results),
                }
                # Prefer enhanced memory service for short-term storage
                if self.enhanced_memory_service:
                    from dova.services.memory_enhanced import MemoryType
                    await self.enhanced_memory_service.store(
                        memory_type=MemoryType.SHORT_TERM,
                        content=memory_content,
                        importance=0.5,
                        user_id=task.user_id,
                        tags=["search", source],
                    )
                elif self.memory_service:
                    await self.remember(
                        content=memory_content,
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
            data = result.data
            # Handle dict response with "papers" key (arxiv-mcp-server format)
            if isinstance(data, dict) and "papers" in data:
                papers = data["papers"]
            elif isinstance(data, list):
                papers = data
            else:
                papers = [data]
            for paper in papers:
                # Skip non-dict items (e.g., error message strings)
                if not isinstance(paper, dict):
                    continue
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

        # Check if user wants "most starred" repos - use sort=stars
        intent = entities.get("intent", {})
        query_lower = query.lower()
        sort_by_stars = any(kw in query_lower for kw in ["most starred", "top repo", "popular", "trending", "best"])

        # Use optimized query from intent if available (removes platform keywords like "github")
        optimized_query = intent.get("search_query", query)

        # Build search query - use topic keywords, removing platform noise
        search_query = self._build_github_query(optimized_query, entities)

        # For noisy queries with filler words, extract the core topic
        # Filler patterns that indicate the query needs simplification
        filler_patterns = ["on with", "most starred", "top repo", "popular", "best", "trending"]
        has_filler = any(fp in search_query.lower() for fp in filler_patterns)

        if has_filler or len(search_query.split()) > 5:
            # Extract meaningful topic phrases from the original query
            topic_patterns = [
                "agentic reasoning", "agentic ai", "llm agent", "ai agent",
                "multi-agent", "reasoning", "chain of thought", "react agent",
                "langchain", "autogen", "crewai", "transformer"
            ]
            found_topics = [tp for tp in topic_patterns if tp in query_lower]
            if found_topics:
                # Use the most specific topic found
                search_query = found_topics[0]
                self._logger.info("github_query_simplified", original=optimized_query, simplified=search_query)

        self._logger.info(
            "github_search",
            original_query=query,
            optimized_query=optimized_query,
            final_query=search_query,
            sort_by_stars=sort_by_stars,
        )

        # Add sort parameter for "most starred" queries
        sort_param = "stars" if sort_by_stars else None
        result = await self.search_github(search_query, search_type="repositories", per_page=20, sort=sort_param)

        if result.success and result.data:
            repos = result.data.get("items", result.data) if isinstance(result.data, dict) else result.data
            if isinstance(repos, list):
                for repo in repos:
                    # Skip non-dict items (e.g., error message strings)
                    if not isinstance(repo, dict):
                        continue
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

        # Build optimized search query using primary_subject
        search_query = self._build_huggingface_query(query, entities)
        self._logger.debug("huggingface_query", original=query, optimized=search_query)

        # Search models
        model_result = await self.search_huggingface(search_query, search_type="models", limit=15)
        if model_result.success and model_result.data:
            models = model_result.data if isinstance(model_result.data, list) else [model_result.data]
            for model in models:
                # Skip non-dict items (e.g., "No models found" message strings)
                if not isinstance(model, dict):
                    continue
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
        dataset_result = await self.search_huggingface(search_query, search_type="datasets", limit=10)
        if dataset_result.success and dataset_result.data:
            datasets = dataset_result.data if isinstance(dataset_result.data, list) else [dataset_result.data]
            for dataset in datasets:
                # Skip non-dict items (e.g., "No datasets found" message strings)
                if not isinstance(dataset, dict):
                    continue
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

    async def _search_web(
        self,
        query: str,
        entities: dict[str, Any],
    ) -> ResearchFindings:
        """Search the web using multi-provider service (Brave, Perplexity, Tavily, DuckDuckGo)."""
        findings = ResearchFindings()

        # Lazy initialization of web search service
        if self._web_search_service is None:
            self._web_search_service = create_web_search_service()

        # Build search query using primary_subject for best results
        primary_subject = entities.get("primary_subject", "")
        search_aspects = entities.get("search_aspects", [])

        if primary_subject:
            # Use primary subject as the core, add aspects if specified
            if search_aspects:
                search_query = f"{primary_subject} {' '.join(search_aspects[:2])}"
            else:
                search_query = primary_subject
        else:
            # Fallback to original query
            search_query = query

        self._logger.debug("web_search", query=search_query, primary_subject=primary_subject)

        try:
            # Perform web search with multi-provider service
            response = await self._web_search_service.search(
                query=search_query,
                max_results=10,
            )

            if response.error:
                self._logger.warning("web_search_error", error=response.error)
                return findings

            for result in response.results:
                findings.web_results.append(
                    SearchResult(
                        source="web",
                        title=result.title,
                        url=result.url,
                        description=result.snippet[:500] if result.snippet else "",
                        metadata={
                            "score": result.score,
                            "published_date": result.published_date or "",
                            "provider": result.source_provider,
                        },
                        relevance_score=result.score,
                    )
                )

            self._logger.info(
                "web_search_complete",
                query=search_query,
                provider=response.provider,
                results_count=len(findings.web_results),
            )

        except Exception as e:
            self._logger.error("web_search_error", error=str(e))

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
        """Search sources based on query type and intent, with smart routing."""
        import asyncio

        # Get intent to determine source selection
        intent = entities.get("intent", {})
        primary_source = intent.get("primary_source")
        query_type = intent.get("query_type", "general")
        recommended_sources = intent.get("recommended_sources", ["web", "arxiv", "github", "huggingface"])
        optimized_query = intent.get("search_query", query)

        self._logger.info(
            "search_all_with_intent",
            query_type=query_type,
            primary_source=primary_source,
            recommended_sources=recommended_sources,
            optimized_query=optimized_query,
        )

        # Build search tasks based on recommended sources (smart routing)
        source_methods = {
            "arxiv": lambda: self._search_arxiv(optimized_query, entities),
            "github": lambda: self._search_github(optimized_query, entities),
            "huggingface": lambda: self._search_huggingface(optimized_query, entities),
            "web": lambda: self._search_web(optimized_query, entities),
        }

        # Only search recommended sources (e.g., biographical queries only search web)
        tasks = []
        source_names = []
        for source in recommended_sources:
            if source in source_methods:
                tasks.append(source_methods[source]())
                source_names.append(source)

        self._logger.info(
            "smart_source_routing",
            query_type=query_type,
            sources_to_search=source_names,
            skipped_sources=[s for s in source_methods if s not in source_names],
        )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Map results to sources
        source_findings = {}
        for name, result in zip(source_names, results):
            if isinstance(result, Exception):
                self._logger.warning("search_partial_failure", source=name, error=str(result))
                source_findings[name] = ResearchFindings()
            elif isinstance(result, ResearchFindings):
                source_findings[name] = result
            else:
                source_findings[name] = ResearchFindings()

        # Merge findings with priority ordering based on recommended sources order
        findings = ResearchFindings()
        priority_order = recommended_sources  # Use recommended order as priority

        self._logger.info("result_priority_order", order=priority_order)

        # Merge in priority order
        for source_name in priority_order:
            sf = source_findings.get(source_name, ResearchFindings())
            findings.papers.extend(sf.papers)
            findings.repositories.extend(sf.repositories)
            findings.models.extend(sf.models)
            findings.datasets.extend(sf.datasets)
            findings.web_results.extend(sf.web_results)

        # If primary source was specified but returned few results, log a warning
        if primary_source:
            primary_findings = source_findings.get(primary_source, ResearchFindings())
            primary_count = (
                len(primary_findings.papers) +
                len(primary_findings.repositories) +
                len(primary_findings.models) +
                len(primary_findings.datasets) +
                len(primary_findings.web_results)
            )
            if primary_count == 0:
                self._logger.warning(
                    "primary_source_no_results",
                    source=primary_source,
                    query=optimized_query,
                    hint="User specifically asked for this source but no results found",
                )

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
        # PRIORITY: Use primary_subject if available (the main thing being searched)
        primary_subject = entities.get("primary_subject", "")
        search_terms = entities.get("search_terms", [])

        if primary_subject:
            # Build query around the primary subject
            parts = [primary_subject]
        elif search_terms:
            # Use extracted search terms
            parts = [" OR ".join(search_terms[:3])]
        else:
            # Fallback to cleaned query
            parts = [self._extract_search_keywords(query)]

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
            "language model": "cs.CL",
        }
        for topic in topics:
            topic_lower = topic.lower()
            for key, category in topic_to_category.items():
                if key in topic_lower:
                    parts.append(f"cat:{category}")
                    break

        return " AND ".join(parts) if len(parts) > 1 else parts[0] if parts else query

    def _build_github_query(self, query: str, entities: dict[str, Any]) -> str:
        """Build optimized GitHub query from entities."""
        intent = entities.get("intent", {})
        optimized_query = intent.get("search_query", query)

        # PRIORITY 1: Use primary_subject (the main entity being searched)
        primary_subject = entities.get("primary_subject", "")
        search_terms = entities.get("search_terms", [])
        technologies = entities.get("technologies", [])

        # Build the core search query
        if primary_subject:
            # Primary subject is the most important - use it as the main search
            core_terms = [primary_subject]
            # Add variations from search_terms that relate to primary subject
            for term in search_terms[:2]:
                if term.lower() != primary_subject.lower() and primary_subject.lower() in term.lower():
                    core_terms.append(term)
            search_query = " OR ".join(core_terms) if len(core_terms) > 1 else primary_subject
        elif search_terms:
            # Use the extracted search terms
            search_query = " ".join(search_terms[:3])
        elif technologies:
            # Use technologies as search terms (often contains model/library names)
            search_query = " ".join(technologies[:3])
        elif optimized_query and optimized_query != query:
            # Use the intent-optimized query (stripped of source keywords)
            search_query = optimized_query
        else:
            # Fallback to cleaned query
            search_query = self._extract_search_keywords(query)

        parts = [search_query]

        # Add language filter only for programming language technologies
        lang_map = {"python": "python", "javascript": "javascript", "typescript": "typescript", "rust": "rust", "go": "go", "java": "java"}
        for tech in technologies:
            tech_lower = tech.lower()
            if tech_lower in lang_map:
                parts.append(f"language:{lang_map[tech_lower]}")
                break

        # Add date filter if time_range is specified
        time_range = entities.get("time_range")
        if time_range:
            parts.append(f"pushed:>{time_range}")

        # Only add minimum stars if not already sorting by stars (avoid over-filtering)
        query_lower = query.lower()
        if not any(kw in query_lower for kw in ["most starred", "top repo", "popular", "trending"]):
            parts.append("stars:>5")

        return " ".join(parts)

    def _build_huggingface_query(self, query: str, entities: dict[str, Any]) -> str:
        """Build optimized HuggingFace query from entities."""
        # PRIORITY: Use primary_subject (the model/dataset name being searched)
        primary_subject = entities.get("primary_subject", "")
        search_terms = entities.get("search_terms", [])
        technologies = entities.get("technologies", [])

        if primary_subject:
            # Primary subject is the best search term for HuggingFace
            return primary_subject
        elif search_terms:
            # Use the first search term (most specific)
            return search_terms[0]
        elif technologies:
            # Technologies often contain model names
            return technologies[0]
        else:
            # Fallback to extracting keywords from query
            return self._extract_search_keywords(query)

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
