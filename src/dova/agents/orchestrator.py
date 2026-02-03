"""
Master Orchestrator Agent for DOVA.

The orchestrator is responsible for:
- Understanding user intent
- Decomposing tasks into sub-tasks
- Dispatching to specialized agents
- Aggregating and synthesizing results
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from dova.agents.base import AgentResult, AgentTask, BaseAgent
from dova.config.providers import LLMRouter, TaskType
from dova.services.collaborative import CollaborativeReasoning, CollaborationMode
from dova.services.evaluation import SelfEvaluator
from dova.services.web_search import WebSearchService, create_web_search_service
from dova.utils.metrics import MetricsCollector

logger = structlog.get_logger(__name__)

# Patterns indicating evaluative queries that benefit from debate analysis
EVALUATIVE_PATTERNS = [
    r"\bevaluate\b", r"\bcompare\b", r"\bvs\.?\b", r"\bversus\b",
    r"\btradeoffs?\b", r"\btrade-offs?\b", r"\bpros?\s+(?:and\s+)?cons?\b",
    r"\badvantages?\s+(?:and\s+)?disadvantages?\b",
    r"\bstrengths?\s+(?:and\s+)?weaknesses?\b",
    r"\bshould\s+(?:i|we)\s+(?:use|choose|pick|adopt)\b",
    r"\bwhich\s+(?:is|are)\s+(?:better|best)\b",
    r"\bdebate\b", r"\barguments?\s+(?:for|against)\b",
]


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
    recommended_sources: list[str] = field(default_factory=list)


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
        web_search_service: WebSearchService | None = None,
    ):
        super().__init__(llm_router, mcp_client, metrics)
        self.agents = agents or {}
        self._task_graph: dict[str, TaskNode] = {}
        self.reasoning_mode = reasoning_mode
        self._collaborative = CollaborativeReasoning(llm_func=self.think)
        self._evaluator = SelfEvaluator(min_confidence=0.6)
        self._web_search_service = web_search_service

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

            # Step 2: Handle general questions directly with LLM
            if intent.intent == UserIntent.GENERAL_QUESTION:
                answer = await self._answer_general_question(query)
                return self._wrap_result(
                    task,
                    True,
                    data={"summary": answer, "intent": "general_question"},
                    execution_time_ms=(time.time() - start_time) * 1000,
                    intent=intent.intent.value,
                    tasks_executed=0,
                )

            # Step 3: Build task graph based on intent
            task_graph = await self._build_task_graph(query, intent, task)

            # Step 4: Execute task graph
            results = await self._execute_task_graph(task_graph)

            # Step 5: Check if we have actual search results
            raw_results = self._collect_raw_results(results)
            has_actual_results = any(
                raw_results.get(k) for k in ["papers", "repositories", "models", "datasets"]
            )

            # If no actual results found, fall back to answering directly with caveats
            if not has_actual_results:
                self._logger.info("no_search_results", query=query)
                answer = await self._answer_general_question(query)
                # Add caveat about no search results
                answer += "\n\n---\n**Note:** No relevant results were found in the research databases (ArXiv, GitHub, HuggingFace). This response is based on general knowledge and may not reflect the most current or specialized information on this topic."
                return self._wrap_result(
                    task,
                    True,
                    data={"summary": answer, "intent": intent.intent.value, "search_results_found": False},
                    execution_time_ms=(time.time() - start_time) * 1000,
                    intent=intent.intent.value,
                    tasks_executed=len(results),
                )

            # Step 6: Synthesize results (only if we have actual data)
            synthesized = await self._synthesize_results(query, intent, results)

            # Step 7: Check if debate should be invoked
            reasoning_mode = task.params.get("reasoning_mode", "standard")
            should_debate = (
                reasoning_mode == "collaborative" or
                self._is_evaluative_query(query)
            )

            if should_debate and "debate" in self.agents:
                debate_result = await self._invoke_debate(
                    query=query,
                    synthesis_context=synthesized,
                    user_id=task.user_id,
                )
                synthesized = self._merge_debate_results(synthesized, debate_result)

            # Step 8: Validate synthesis quality
            if synthesized.get("summary"):
                evaluation = await self._evaluator.evaluate(
                    response=str(synthesized.get("summary", "")),
                    prompt=query,
                )
                if evaluation.confidence < 0.5:
                    synthesized["quality_warning"] = "This synthesis may have quality issues. Please verify the information from primary sources."
                    self._logger.warning("low_synthesis_quality", confidence=evaluation.confidence)

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
        from datetime import datetime
        current_date = datetime.now().strftime("%B %Y")  # e.g., "February 2026"
        current_year = datetime.now().year

        classification_prompt = f"""Analyze this research query and extract structured information.

Query: "{query}"
Current Date: {current_date}

CRITICAL INSTRUCTIONS:
1. "primary_subject" MUST be the main entity/topic being searched (e.g., a model name like "qwen3", "GPT-4", "llama", or a specific concept)
2. Extract the EXACT names/terms from the query - do NOT generalize or paraphrase
3. If the query mentions a specific model, library, or project name, that MUST be in primary_subject AND search_terms
4. For recent/news queries, if adding years to search_terms, use CURRENT year ({current_year}) and recent past years - NEVER use outdated years

Respond with JSON:
{{
    "intent": "<one of: research_query, code_search, paper_search, model_search, innovation_request, validation_request, general_question>",
    "confidence": <0.0-1.0>,
    "entities": {{
        "primary_subject": "<THE MAIN THING being searched - model name, project name, or core concept - EXTRACT EXACTLY from query>",
        "search_terms": ["<EXACT keywords to use in searches - include primary_subject variations>"],
        "topics": ["<broader topic areas>"],
        "technologies": ["<specific technologies, models, frameworks mentioned>"],
        "authors": ["<author names if mentioned>"],
        "repositories": ["<repo names if mentioned>"],
        "time_range": "<if specified, e.g., 'last 6 months'>",
        "constraints": ["<other constraints>"],
        "search_aspects": ["<specific aspects requested: performance, architecture, implementation, etc.>"]
    }},
    "requires_profiling": <true if personalization would help>,
    "requires_validation": <true if code validation is requested>,
    "recommended_sources": ["<appropriate sources: web, arxiv, github, huggingface>"]
}}

IMPORTANT SOURCE SELECTION RULES:
- "web": For news, current events, recent announcements, politics, general knowledge questions
- "arxiv": For academic papers, research, scientific studies
- "github": For code, implementations, software projects, technical libraries
- "huggingface": For ML models, datasets, AI/ML specific topics
- DO NOT recommend github/huggingface for news, politics, current events, or non-technical topics

Example for "Lookup qwen3-max-thinking architectures":
{{
    "intent": "model_search",
    "confidence": 0.95,
    "entities": {{
        "primary_subject": "qwen3-max-thinking",
        "search_terms": ["qwen3", "qwen-3", "qwen3-max-thinking", "qwen max thinking"],
        "topics": ["model architecture", "language models"],
        "technologies": ["qwen3-max-thinking", "qwen", "transformer"],
        ...
    }}
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
                recommended_sources=parsed.get("recommended_sources", ["web"]),
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
        requested_sources = parent_task.params.get("sources", ["arxiv", "github", "huggingface", "web"])

        # Use LLM-recommended sources to filter inappropriate ones
        recommended = intent.recommended_sources if intent.recommended_sources else requested_sources
        # Only search sources that are both requested AND recommended (intelligent filtering)
        smart_sources = [s for s in requested_sources if s in recommended] or recommended

        self._logger.debug(
            "source_selection",
            requested=requested_sources,
            recommended=recommended,
            selected=smart_sources,
        )

        # Filter to only configured sources (check MCP registry or web search service)
        available_sources = []
        for source in smart_sources:
            if source == "web":
                # Web search is always available (DuckDuckGo fallback)
                available_sources.append(source)
            elif self.mcp_client:
                server = self.mcp_client.registry.get_server(source)
                if server:
                    available_sources.append(source)

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

            # Web search (only if configured with Tavily)
            if "web" in available_sources:
                graph["web_search"] = TaskNode(
                    id="web_search",
                    agent_type="research",
                    task=AgentTask(
                        type="search",
                        params={
                            "source": "web",
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

    async def _answer_general_question(self, query: str) -> str:
        """Answer a general question with quality validation and optional search enrichment."""

        # Step 1: First check if the topic requires recent/specialized knowledge
        confidence_check_prompt = f"""Assess your knowledge about this topic and respond in JSON:

Question: {query}

Respond with JSON only:
{{
    "confidence": <0.0-1.0, how confident are you about this topic>,
    "needs_search": <true/false, whether this requires recent or specialized information>,
    "reason": "<brief reason for your assessment>"
}}"""

        try:
            confidence_response = await self.think(confidence_check_prompt)
            import json
            try:
                confidence_data = json.loads(confidence_response.strip())
                confidence = confidence_data.get("confidence", 0.5)
                needs_search = confidence_data.get("needs_search", False)
            except json.JSONDecodeError:
                confidence = 0.5
                needs_search = True

            self._logger.debug(
                "confidence_assessment",
                confidence=confidence,
                needs_search=needs_search,
            )

            # Step 2: If low confidence or needs search, search for context
            search_context = ""
            if confidence < 0.7 or needs_search:
                search_context = await self._search_for_context(query, query_type="general_question")

            # Step 3: Generate the answer with context if available
            if search_context:
                prompt = f"""You are DOVA, a knowledgeable AI research assistant.

Answer the following question using both your knowledge AND the search results provided.

Question: {query}

Search Results:
{search_context}

Instructions:
1. Provide a comprehensive but concise answer
2. Cite specific sources from the search results when applicable
3. Clearly distinguish between verified information (from search) and general knowledge
4. If the search results don't contain relevant information, acknowledge this
5. If you're uncertain about something, say so explicitly"""
            else:
                prompt = f"""You are DOVA, a knowledgeable AI research assistant.

Answer the following question:

Question: {query}

Instructions:
1. Provide a comprehensive but concise answer
2. If this topic relates to very recent developments or specialized research, acknowledge that your knowledge may be limited
3. If you're uncertain about specific details, say so explicitly
4. Suggest searching for more recent information if the topic likely requires it"""

            response = await self.think(prompt)
            answer = response.strip()

            # Step 4: Evaluate the response quality
            evaluation = await self._evaluator.evaluate(
                response=answer,
                prompt=query,
                expected_format=None,
            )

            self._logger.debug(
                "answer_evaluated",
                confidence=evaluation.confidence,
                caveats=evaluation.caveats,
            )

            # Step 5: Add quality warnings if needed
            if evaluation.confidence < 0.6:
                warning = "\n\n---\n**Note:** This response may have limitations. Consider searching for more specific or recent information on this topic."
                answer += warning
            elif evaluation.caveats:
                caveat_text = "; ".join(evaluation.caveats)
                answer += f"\n\n---\n**Note:** {caveat_text}"

            return answer

        except Exception as e:
            self._logger.error("general_question_error", error=str(e))
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"

    async def _search_for_context(self, query: str, query_type: str = "general") -> str:
        """Search appropriate sources for context to enrich the answer."""
        context_parts = []

        # Determine which sources are appropriate for this query type
        sources_to_search = self._select_appropriate_sources(query, query_type)
        self._logger.debug("context_sources_selected", sources=sources_to_search, query_type=query_type)

        # PRIORITY 1: Web search for current events, news, and general questions
        if "web" in sources_to_search:
            web_context = await self._search_web_for_context(query)
            if web_context:
                context_parts.append(web_context)

        # Only search technical sources if appropriate for the query
        if not self.mcp_client:
            return "\n\n".join(context_parts) if context_parts else ""

        try:
            for server_name in sources_to_search:
                if server_name == "web":
                    continue  # Already handled above

                server = self.mcp_client.registry.get_server(server_name)
                if not server:
                    continue

                try:
                    if server_name == "arxiv":
                        result = await self.mcp_client.invoke(
                            server_name, "search_papers",
                            {"query": query, "max_results": 3}
                        )
                        if result:
                            result_str = str(result)[:2000]
                            context_parts.append(f"**ArXiv Papers:**\n{result_str}")

                    elif server_name == "github":
                        result = await self.mcp_client.invoke(
                            server_name, "search_repositories",
                            {"query": query, "per_page": 3}
                        )
                        if result:
                            result_str = str(result)[:2000]
                            context_parts.append(f"**GitHub Repositories:**\n{result_str}")

                    elif server_name == "huggingface":
                        result = await self.mcp_client.invoke(
                            server_name, "model_search",
                            {"query": query, "limit": 3}
                        )
                        if result:
                            result_str = str(result)[:2000]
                            context_parts.append(f"**HuggingFace Models:**\n{result_str}")

                except Exception as e:
                    self._logger.debug(
                        "context_search_failed",
                        server=server_name,
                        error=str(e),
                    )
                    continue

        except Exception as e:
            self._logger.warning("context_search_error", error=str(e))

        return "\n\n".join(context_parts) if context_parts else ""

    def _select_appropriate_sources(self, query: str, query_type: str) -> list[str]:
        """Select appropriate sources based on query content and type."""
        query_lower = query.lower()

        # Indicators for different query types
        news_indicators = [
            "news", "latest", "recent", "today", "yesterday", "announced",
            "nominate", "nominated", "election", "president", "congress",
            "government", "policy", "politics", "trump", "biden", "fed",
            "chairman", "secretary", "minister", "breaking", "update"
        ]
        technical_indicators = [
            "code", "implementation", "library", "framework", "api",
            "model", "architecture", "algorithm", "benchmark", "performance",
            "github", "repository", "huggingface", "arxiv", "paper",
            "transformer", "neural", "training", "dataset"
        ]
        research_indicators = [
            "research", "study", "paper", "academic", "journal",
            "arxiv", "publication", "findings", "methodology"
        ]

        # Check for news/current events
        is_news_query = any(indicator in query_lower for indicator in news_indicators)
        is_technical_query = any(indicator in query_lower for indicator in technical_indicators)
        is_research_query = any(indicator in query_lower for indicator in research_indicators)

        sources = []

        # Always include web for news and current events
        if is_news_query or query_type == "general_question":
            sources.append("web")

        # Only add technical sources if the query is technical
        if is_technical_query or is_research_query:
            if is_research_query:
                sources.append("arxiv")
            sources.append("github")
            sources.append("huggingface")

        # If nothing matched, default to web search
        if not sources:
            sources.append("web")

        return sources

    async def _search_web_for_context(self, query: str) -> str:
        """Search the web using multi-provider service for current information."""
        try:
            # Lazy initialization of web search service
            if self._web_search_service is None:
                self._web_search_service = create_web_search_service()

            self._logger.debug("web_context_search", query=query)

            response = await self._web_search_service.search(
                query=query,
                max_results=5,
            )

            if response.error or not response.results:
                self._logger.debug("web_search_no_results", error=response.error)
                return ""

            # Format results for context
            formatted = [f"**Web Search Results (via {response.provider}):**"]
            for i, result in enumerate(response.results[:5], 1):
                title = result.title or "No title"
                url = result.url or ""
                content = result.snippet[:300] if result.snippet else ""
                date_str = f" ({result.published_date})" if result.published_date else ""
                formatted.append(f"\n{i}. **{title}**{date_str}\n   URL: {url}\n   {content}...")

            return "\n".join(formatted)

        except Exception as e:
            self._logger.warning("web_context_error", error=str(e))
            return ""

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

        # Return raw results if no synthesis available
        has_results = any(
            raw_results.get(k) for k in ["papers", "repositories", "models", "datasets"]
        )

        if has_results:
            # Generate a simple summary if we have results but no synthesis
            summary = f"Found {len(raw_results.get('papers', []))} papers, {len(raw_results.get('repositories', []))} repositories, {len(raw_results.get('models', []))} models."
            return {
                "summary": summary,
                "papers": raw_results.get("papers", []),
                "repositories": raw_results.get("repositories", []),
                "models": raw_results.get("models", []),
                "datasets": raw_results.get("datasets", []),
            }

        # Return empty structure if no results
        return {
            "summary": "No results found for the query.",
            "papers": [],
            "repositories": [],
            "models": [],
            "datasets": [],
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

    def _is_evaluative_query(self, query: str) -> bool:
        """Check if query requires evaluative/debate analysis."""
        query_lower = query.lower()
        return any(re.search(p, query_lower) for p in EVALUATIVE_PATTERNS)

    async def _invoke_debate(
        self,
        query: str,
        synthesis_context: dict[str, Any],
        user_id: str | None,
    ) -> dict[str, Any]:
        """Invoke debate agent with synthesis context."""
        debate_agent = self.agents.get("debate")
        if not debate_agent:
            return {}

        self._logger.info("debate_starting", query=query)

        context = {
            "summary": synthesis_context.get("summary", "")[:1000],
            "key_findings": [f.get("title", "") for f in synthesis_context.get("insights", [])[:5]],
        }

        task = AgentTask(
            type="debate",
            params={"topic": query, "context": context},
            user_id=user_id,
        )

        result = await debate_agent.execute(task)
        return result.data if result.success else {}

    def _merge_debate_results(
        self,
        synthesized: dict[str, Any],
        debate_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge debate results into synthesis output."""
        if not debate_result:
            return synthesized

        return {
            **synthesized,
            "debate": {
                "summary": debate_result.get("summary", ""),
                "bull_strengths": debate_result.get("bull_strengths", []),
                "bear_concerns": debate_result.get("bear_concerns", []),
                "recommendation": debate_result.get("recommendation", ""),
                "confidence": debate_result.get("confidence_score", 0.5),
            },
        }

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
