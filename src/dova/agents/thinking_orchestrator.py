"""
Thinking Orchestrator for DOVA.

A deliberation-first orchestrator that reasons about user needs
before deciding whether tools are needed. Unlike the task-graph
based DOVAOrchestrator, this one:

1. Deliberates first - reasons about what user actually needs
2. Models users deeply - considers expertise, style, goals
3. Treats tools as optional - only invokes them when reasoning
   decides they would genuinely help
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from dova.agents.base import AgentResult, AgentTask, BaseAgent
from dova.agents.conversation_context import ConversationContext, ConversationTurn
from dova.agents.user_model import ExpertiseLevel, ResponseDepth, UserModel
from dova.config.providers import LLMRouter, TaskType
from dova.services.web_search import WebSearchService, create_web_search_service
from dova.utils.metrics import MetricsCollector

logger = structlog.get_logger(__name__)


class ActionDecision(Enum):
    """Decisions the orchestrator can make after deliberation."""

    RESPOND_DIRECTLY = "respond_directly"  # Answer from context/knowledge
    USE_TOOLS = "use_tools"  # Need to search external sources
    CLARIFY = "clarify"  # Need more information from user


@dataclass
class ToolConsideration:
    """Explicit reasoning about whether a tool would help."""

    tool_name: str
    would_help: bool
    rationale: str
    search_query: str = ""


@dataclass
class Deliberation:
    """Result of the deliberation process."""

    understanding: str  # What user actually needs
    can_answer_from_context: bool
    knowledge_gaps: list[str] = field(default_factory=list)
    tools_to_use: list[ToolConsideration] = field(default_factory=list)
    action: ActionDecision = ActionDecision.RESPOND_DIRECTLY
    reasoning: str = ""
    clarification_needed: str = ""


# Deliberation prompt template
DELIBERATION_PROMPT = """You are deciding how to help a user. Think carefully before acting.

USER QUERY: {query}

ABOUT THIS USER:
- Expertise: {expertise_areas}
- Communication style: {communication_style}
- Session goals: {session_goals}

CONVERSATION CONTEXT:
- Topic: {current_topic}
- Already discussed: {entities_discussed}
- Recent turns: {recent_turns}

AVAILABLE TOOLS (use ONLY if genuinely needed):
- arxiv: Academic papers (use for: research papers, scientific studies, technical methods)
- github: Code repositories (use for: implementations, libraries, code examples)
- huggingface: ML models/datasets (use for: pretrained models, datasets, ML-specific)
- web: Web search (use for: news, current events, general information, non-technical topics)

THINK THROUGH:
1. What does the user ACTUALLY need? (Not just what they literally asked)
2. Can I answer from existing context, conversation history, or general knowledge?
3. If tools are needed, which specific ones and why? (Don't use tools unnecessarily)
4. Is this a follow-up about something already discussed? (Use context if so)

IMPORTANT:
- For follow-up questions about papers/repos/models already discussed, DO NOT search again - use the context
- For general knowledge questions, prefer responding directly unless recent/specialized info is needed
- For news/current events, use web search
- Only use arxiv/github/huggingface for technical research needs

Respond with JSON only:
{{
    "understanding": "what user actually needs",
    "can_answer_from_context": true or false,
    "knowledge_gaps": ["what I don't know that would help"],
    "tools_to_use": [
        {{"tool": "arxiv|github|huggingface|web", "rationale": "why this tool", "query": "search query"}}
    ],
    "action": "respond_directly|use_tools|clarify",
    "reasoning": "why this decision",
    "clarification_needed": "question for user if action is clarify"
}}"""


class ThinkingOrchestrator(BaseAgent):
    """
    Deliberation-first orchestrator for DOVA.

    Instead of building a task graph and executing all sources,
    this orchestrator:
    1. Loads user context and conversation history
    2. Deliberates about what user actually needs
    3. Decides whether tools are needed (and which ones)
    4. Executes only necessary tools
    5. Personalizes response based on user model
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        agents: dict[str, BaseAgent] | None = None,
        mcp_client: Any | None = None,
        metrics: MetricsCollector | None = None,
        web_search_service: WebSearchService | None = None,
        memory_service: Any | None = None,
    ):
        super().__init__(llm_router, mcp_client, metrics, memory_service=memory_service)
        self.agents = agents or {}
        self._web_search_service = web_search_service
        self._user_models: dict[str, UserModel] = {}
        self._contexts: dict[str, ConversationContext] = {}

    @property
    def system_prompt(self) -> str:
        return """You are DOVA's Thinking Orchestrator, an AI that carefully reasons about
user needs before taking action. You prioritize understanding over doing.

Your approach:
1. First understand what the user truly needs
2. Consider what you already know from context
3. Only use tools when they would genuinely help
4. Personalize responses based on user expertise and style

You never search blindly - you always have a reason for each action."""

    async def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute with deliberation-first approach.

        Args:
            task: Task containing the user query

        Returns:
            AgentResult with personalized response
        """
        start_time = time.time()

        try:
            query = task.params.get("query", "")
            if not query:
                return self._wrap_result(task, False, error="No query provided")

            session_id = task.params.get("session_id") or task.id
            self._logger.info(
                "thinking_orchestrator_starting",
                query=query,
                user_id=task.user_id,
                session_id=session_id,
            )

            # 1. Load user model and conversation context
            user_model = await self._load_user_model(task.user_id)
            context = await self._load_conversation_context(session_id, task.user_id)

            # Add user query to context
            context.add_turn(role="user", content=query)

            # 2. DELIBERATE - the key innovation
            deliberation = await self._deliberate(query, user_model, context)

            self._logger.info(
                "deliberation_complete",
                action=deliberation.action.value,
                tools_count=len(deliberation.tools_to_use),
                can_answer_from_context=deliberation.can_answer_from_context,
            )

            # 3. Execute based on deliberation decision
            response: str
            tools_used: list[str] = []
            action_result: dict[str, Any] | None = None

            if deliberation.action == ActionDecision.RESPOND_DIRECTLY:
                response = await self._respond_from_context(
                    query, deliberation, user_model, context
                )
            elif deliberation.action == ActionDecision.USE_TOOLS:
                tool_results = await self._execute_selected_tools(deliberation)
                tools_used = [t.tool_name for t in deliberation.tools_to_use if t.would_help]
                action_result = tool_results
                response = await self._synthesize_with_results(
                    query, tool_results, user_model, context, deliberation
                )
            else:  # CLARIFY
                response = deliberation.clarification_needed
                context.last_assistant_question = response

            # 4. Update context with assistant response
            context.add_turn(
                role="assistant",
                content=response,
                tools=tools_used,
                action=deliberation.action.value,
                rationale=deliberation.reasoning,
            )

            # Extract entities from results for future reference
            if action_result:
                self._update_context_entities(context, action_result)

            # Save updated context
            await self._save_conversation_context(context)

            execution_time = (time.time() - start_time) * 1000

            return self._wrap_result(
                task,
                True,
                data={
                    "response": response,
                    "deliberation": {
                        "action": deliberation.action.value,
                        "reasoning": deliberation.reasoning,
                        "tools_used": tools_used,
                    },
                    "action_result": action_result,
                },
                execution_time_ms=execution_time,
            )

        except Exception as e:
            self._logger.exception("thinking_orchestrator_error", error=str(e))
            return self._wrap_result(
                task,
                False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _deliberate(
        self,
        query: str,
        user_model: UserModel,
        context: ConversationContext,
    ) -> Deliberation:
        """
        Reason about what user needs and decide on action.

        This is the core innovation - explicit reasoning before action.
        """
        # Format user model info
        expertise_str = ", ".join(
            f"{topic}: {level.value}"
            for topic, level in list(user_model.expertise_areas.items())[:5]
        ) or "unknown"

        style_str = f"{user_model.formality}, prefers {'code examples' if user_model.prefers_code_examples else 'explanations'}"
        goals_str = ", ".join(user_model.current_goals[:3]) or "not specified"

        # Format context info
        entities_discussed = []
        if context.papers_discussed:
            entities_discussed.extend([f"paper: {p.get('title', '')[:50]}" for p in context.papers_discussed[:3]])
        if context.repos_discussed:
            entities_discussed.extend([f"repo: {r.get('name', '')}" for r in context.repos_discussed[:3]])
        if context.models_discussed:
            entities_discussed.extend([f"model: {m.get('id', '')}" for m in context.models_discussed[:3]])

        entities_str = ", ".join(entities_discussed) or "none"

        recent_turns = context.get_conversation_summary(500)

        # Build and execute deliberation prompt
        prompt = DELIBERATION_PROMPT.format(
            query=query,
            expertise_areas=expertise_str,
            communication_style=style_str,
            session_goals=goals_str,
            current_topic=context.current_topic or "none",
            entities_discussed=entities_str,
            recent_turns=recent_turns or "none",
        )

        response = await self.think(
            prompt,
            task_type=TaskType.REASONING,
            temperature=0.3,
        )

        # Parse deliberation response
        return self._parse_deliberation(response)

    def _parse_deliberation(self, response: str) -> Deliberation:
        """Parse the LLM's deliberation response into structured form."""
        try:
            # Extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            data = json.loads(response.strip())

            # Parse tools
            tools = []
            for tool_data in data.get("tools_to_use", []):
                tools.append(ToolConsideration(
                    tool_name=tool_data.get("tool", ""),
                    would_help=True,
                    rationale=tool_data.get("rationale", ""),
                    search_query=tool_data.get("query", ""),
                ))

            # Map action string to enum
            action_str = data.get("action", "respond_directly")
            try:
                action = ActionDecision(action_str)
            except ValueError:
                action = ActionDecision.RESPOND_DIRECTLY

            return Deliberation(
                understanding=data.get("understanding", ""),
                can_answer_from_context=data.get("can_answer_from_context", True),
                knowledge_gaps=data.get("knowledge_gaps", []),
                tools_to_use=tools,
                action=action,
                reasoning=data.get("reasoning", ""),
                clarification_needed=data.get("clarification_needed", ""),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self._logger.warning("deliberation_parse_error", error=str(e), response=response[:200])
            return Deliberation(
                understanding="Unable to parse deliberation",
                can_answer_from_context=False,
                action=ActionDecision.RESPOND_DIRECTLY,
                reasoning="Parse error, defaulting to direct response",
            )

    async def _respond_from_context(
        self,
        query: str,
        deliberation: Deliberation,
        user_model: UserModel,
        context: ConversationContext,
    ) -> str:
        """Generate response using existing context without tools."""
        # Build context for response
        context_parts = [f"User question: {query}"]

        if deliberation.understanding:
            context_parts.append(f"Understanding: {deliberation.understanding}")

        # Add relevant entities from context
        if context.papers_discussed:
            papers_info = []
            for p in context.papers_discussed[:5]:
                info = f"- {p.get('title', 'Unknown')}"
                if p.get('authors'):
                    authors = p['authors']
                    if isinstance(authors, list):
                        info += f" by {', '.join(str(a) for a in authors[:3])}"
                    else:
                        info += f" by {authors}"
                papers_info.append(info)
            context_parts.append("Papers discussed:\n" + "\n".join(papers_info))

        if context.repos_discussed:
            repos_info = [f"- {r.get('name', '')}: {r.get('description', '')[:100]}" for r in context.repos_discussed[:3]]
            context_parts.append("Repos discussed:\n" + "\n".join(repos_info))

        if context.models_discussed:
            models_info = [f"- {m.get('id', '')}" for m in context.models_discussed[:3]]
            context_parts.append("Models discussed:\n" + "\n".join(models_info))

        # Add conversation history
        if len(context.turns) > 1:
            context_parts.append(f"Recent conversation:\n{context.get_conversation_summary(400)}")

        # Personalization based on user model
        style_instructions = self._get_style_instructions(user_model)

        prompt = f"""Generate a helpful response based on this context:

{chr(10).join(context_parts)}

{style_instructions}

Response:"""

        return await self.think(
            prompt,
            task_type=TaskType.CHAT,
            temperature=0.5,
        )

    async def _execute_selected_tools(self, deliberation: Deliberation) -> dict[str, Any]:
        """Execute only the tools that deliberation decided are needed."""
        results: dict[str, Any] = {
            "papers": [],
            "repositories": [],
            "models": [],
            "datasets": [],
            "web_results": [],
        }

        for tool in deliberation.tools_to_use:
            if not tool.would_help:
                continue

            try:
                self._logger.info(
                    "executing_tool",
                    tool=tool.tool_name,
                    query=tool.search_query,
                    rationale=tool.rationale,
                )

                if tool.tool_name == "arxiv":
                    arxiv_results = await self._search_arxiv(tool.search_query)
                    results["papers"].extend(arxiv_results)

                elif tool.tool_name == "github":
                    github_results = await self._search_github(tool.search_query)
                    results["repositories"].extend(github_results)

                elif tool.tool_name == "huggingface":
                    hf_results = await self._search_huggingface(tool.search_query)
                    results["models"].extend(hf_results.get("models", []))
                    results["datasets"].extend(hf_results.get("datasets", []))

                elif tool.tool_name == "web":
                    web_results = await self._search_web(tool.search_query)
                    results["web_results"].extend(web_results)

            except Exception as e:
                self._logger.warning("tool_execution_error", tool=tool.tool_name, error=str(e))

        return results

    async def _search_arxiv(self, query: str) -> list[dict[str, Any]]:
        """Search ArXiv for papers."""
        result = await self.search_arxiv(query, max_results=10)
        if not result.success or not result.data:
            return []

        data = result.data
        if isinstance(data, dict) and "papers" in data:
            papers = data["papers"]
        elif isinstance(data, list):
            papers = data
        else:
            papers = [data]

        return [
            {
                "title": p.get("title", ""),
                "url": p.get("url", p.get("id", "")),
                "description": p.get("summary", p.get("abstract", ""))[:500],
                "authors": p.get("authors", []),
                "arxiv_id": p.get("id", ""),
                "published": p.get("published", ""),
            }
            for p in papers if isinstance(p, dict)
        ]

    async def _search_github(self, query: str) -> list[dict[str, Any]]:
        """Search GitHub for repositories."""
        result = await self.search_github(query, search_type="repositories", per_page=10)
        if not result.success or not result.data:
            return []

        repos = result.data.get("items", result.data) if isinstance(result.data, dict) else result.data
        if not isinstance(repos, list):
            return []

        return [
            {
                "name": r.get("full_name", r.get("name", "")),
                "url": r.get("html_url", r.get("url", "")),
                "description": r.get("description", "")[:300],
                "stars": r.get("stargazers_count", 0),
                "language": r.get("language", ""),
            }
            for r in repos if isinstance(r, dict)
        ]

    async def _search_huggingface(self, query: str) -> dict[str, list[dict[str, Any]]]:
        """Search HuggingFace for models and datasets."""
        results: dict[str, list[dict[str, Any]]] = {"models": [], "datasets": []}

        # Search models
        model_result = await self.search_huggingface(query, search_type="models", limit=10)
        if model_result.success and model_result.data:
            models = model_result.data if isinstance(model_result.data, list) else [model_result.data]
            results["models"] = [
                {
                    "id": m.get("id", m.get("modelId", "")),
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                    "tags": m.get("tags", []),
                }
                for m in models if isinstance(m, dict)
            ]

        # Search datasets
        dataset_result = await self.search_huggingface(query, search_type="datasets", limit=5)
        if dataset_result.success and dataset_result.data:
            datasets = dataset_result.data if isinstance(dataset_result.data, list) else [dataset_result.data]
            results["datasets"] = [
                {
                    "id": d.get("id", d.get("datasetId", "")),
                    "downloads": d.get("downloads", 0),
                }
                for d in datasets if isinstance(d, dict)
            ]

        return results

    async def _search_web(self, query: str) -> list[dict[str, Any]]:
        """Search web using multi-provider service."""
        if self._web_search_service is None:
            self._web_search_service = create_web_search_service()

        try:
            response = await self._web_search_service.search(query=query, max_results=10)
            if response.error:
                return []

            return [
                {
                    "title": r.title,
                    "url": r.url,
                    "description": r.snippet[:500] if r.snippet else "",
                    "published_date": r.published_date,
                }
                for r in response.results
            ]
        except Exception as e:
            self._logger.warning("web_search_error", error=str(e))
            return []

    async def _synthesize_with_results(
        self,
        query: str,
        results: dict[str, Any],
        user_model: UserModel,
        context: ConversationContext,
        deliberation: Deliberation,
    ) -> str:
        """Synthesize results into a personalized response."""
        # Build result summary
        result_parts = []

        if results.get("papers"):
            papers = results["papers"][:5]
            paper_lines = [f"- {p.get('title', 'Unknown')}" for p in papers]
            result_parts.append(f"**Papers Found ({len(results['papers'])}):**\n" + "\n".join(paper_lines))

        if results.get("repositories"):
            repos = results["repositories"][:5]
            repo_lines = [f"- {r.get('name', '')} ({r.get('stars', 0)} stars): {r.get('description', '')[:80]}" for r in repos]
            result_parts.append(f"**Repositories Found ({len(results['repositories'])}):**\n" + "\n".join(repo_lines))

        if results.get("models"):
            models = results["models"][:3]
            model_lines = [f"- {m.get('id', '')} ({m.get('downloads', 0)} downloads)" for m in models]
            result_parts.append(f"**Models Found ({len(results['models'])}):**\n" + "\n".join(model_lines))

        if results.get("web_results"):
            web = results["web_results"][:5]
            web_lines = [f"- {w.get('title', '')}: {w.get('description', '')[:100]}" for w in web]
            result_parts.append(f"**Web Results ({len(results['web_results'])}):**\n" + "\n".join(web_lines))

        # Get personalization
        style_instructions = self._get_style_instructions(user_model)

        prompt = f"""Synthesize these search results into a helpful response.

User Question: {query}
Understanding: {deliberation.understanding}

Search Results:
{chr(10).join(result_parts) if result_parts else "No results found."}

{style_instructions}

Guidelines:
- Be direct and informative
- Reference specific findings with context
- Acknowledge limitations if results are sparse
- Suggest follow-up if relevant

Response:"""

        return await self.think(
            prompt,
            task_type=TaskType.SUMMARIZATION,
            temperature=0.5,
        )

    def _get_style_instructions(self, user_model: UserModel) -> str:
        """Get personalization instructions based on user model."""
        instructions = []

        # Depth
        if user_model.preferred_depth == ResponseDepth.BRIEF:
            instructions.append("Keep the response concise and to the point.")
        elif user_model.preferred_depth == ResponseDepth.DETAILED:
            instructions.append("Provide detailed explanations with context.")

        # Style
        if user_model.formality == "casual":
            instructions.append("Use a conversational tone.")
        elif user_model.formality == "formal":
            instructions.append("Use a professional, formal tone.")

        # Code examples
        if user_model.prefers_code_examples:
            instructions.append("Include code examples where relevant.")

        # Citations
        if user_model.prefers_citations:
            instructions.append("Cite sources and provide links when available.")

        # Expertise
        expert_areas = [t for t, l in user_model.expertise_areas.items() if l == ExpertiseLevel.EXPERT]
        if expert_areas:
            instructions.append(f"The user is an expert in: {', '.join(expert_areas[:3])}. Use technical terminology appropriately.")

        beginner_areas = [t for t, l in user_model.expertise_areas.items() if l == ExpertiseLevel.BEGINNER]
        if beginner_areas:
            instructions.append(f"The user is a beginner in: {', '.join(beginner_areas[:3])}. Explain concepts clearly.")

        return "Style: " + " ".join(instructions) if instructions else ""

    def _update_context_entities(self, context: ConversationContext, results: dict[str, Any]) -> None:
        """Update conversation context with entities from search results."""
        for paper in results.get("papers", [])[:5]:
            context.add_paper(paper)

        for repo in results.get("repositories", [])[:5]:
            context.add_repo(repo)

        for model in results.get("models", [])[:3]:
            context.add_model(model)

    async def _load_user_model(self, user_id: str | None) -> UserModel:
        """Load or create user model."""
        if not user_id:
            return UserModel(user_id="anonymous")

        # Check in-memory cache
        if user_id in self._user_models:
            return self._user_models[user_id]

        # Try to load from profiling agent if available
        profiling_agent = self.agents.get("profiling")
        if profiling_agent and hasattr(profiling_agent, "to_user_model"):
            try:
                user_model = await profiling_agent.to_user_model(user_id)
                self._user_models[user_id] = user_model
                return user_model
            except Exception as e:
                self._logger.warning("user_model_load_error", error=str(e))

        # Create new user model
        user_model = UserModel(user_id=user_id)
        self._user_models[user_id] = user_model
        return user_model

    async def _load_conversation_context(
        self,
        session_id: str,
        user_id: str | None,
    ) -> ConversationContext:
        """Load or create conversation context."""
        if session_id in self._contexts:
            return self._contexts[session_id]

        # Try to load from memory service
        if self.memory_service:
            try:
                from dova.services.memory_enhanced import MemoryType

                results = await self.memory_service.search_by_tags(
                    tags=[f"session:{session_id}"],
                    user_id=user_id,
                    top_k=1,
                )
                if results and hasattr(results[0], "content"):
                    ctx = ConversationContext.from_dict(results[0].content)
                    self._contexts[session_id] = ctx
                    return ctx
            except Exception as e:
                self._logger.debug("context_load_error", error=str(e))

        # Create new context
        ctx = ConversationContext(session_id=session_id, user_id=user_id)
        self._contexts[session_id] = ctx
        return ctx

    async def _save_conversation_context(self, context: ConversationContext) -> None:
        """Save conversation context to memory."""
        self._contexts[context.session_id] = context

        if self.memory_service:
            try:
                from dova.services.memory_enhanced import MemoryType

                await self.memory_service.store(
                    memory_type=MemoryType.SHORT_TERM,
                    content=context.to_dict(),
                    user_id=context.user_id,
                    tags=[f"session:{context.session_id}", "conversation_context"],
                )
            except Exception as e:
                self._logger.debug("context_save_error", error=str(e))

    def register_agent(self, agent_type: str, agent: BaseAgent) -> None:
        """Register a specialized agent."""
        self.agents[agent_type] = agent
        self._logger.info("agent_registered", agent_type=agent_type)
