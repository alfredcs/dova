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
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from dova.agents.base import AgentResult, AgentTask, BaseAgent
from dova.agents.conversation_context import ConversationContext, ConversationTurn
from dova.agents.user_model import ExpertiseLevel, ResponseDepth, UserModel
from dova.config.mcp_servers import list_mcp_servers
from dova.config.providers import LLMRouter, TaskType
from dova.services.web_search import (
    ParallelWebSearchService,
    create_parallel_search_service,
)
from dova.utils.metrics import MetricsCollector

logger = structlog.get_logger(__name__)


# Tool descriptions for known tool types
TOOL_DESCRIPTIONS: dict[str, str] = {
    "arxiv": "Academic papers (use for: research papers, scientific studies, technical methods)",
    "github": "Code repositories (use for: implementations, libraries, code examples)",
    "huggingface": "ML models/datasets (use for: pretrained models, datasets, ML-specific)",
    "hugging-face": "ML models/datasets (use for: pretrained models, datasets, ML-specific)",
    "web": "Web search (use for: news, current events, general information, non-technical topics)",
    "image": "Image generation (use for: creating images, visualizations, artwork, illustrations)",
    "awslabs": "AWS services (use for: AWS pricing, documentation, CDK, CloudFormation, Bedrock, etc.)",
}


def get_available_tools() -> dict[str, str]:
    """
    Load available tools from ~/.dova.json and aggregate prefixed tools.

    Tools with names like "awslabs.xyz" are aggregated into "awslabs".
    Returns a dict of tool_name -> description.
    """
    tools: dict[str, str] = {}

    # Always include built-in tools
    for name in ["arxiv", "github", "huggingface", "web", "image"]:
        tools[name] = TOOL_DESCRIPTIONS.get(name, f"{name} search")

    # Load MCP servers from config
    mcp_servers = list_mcp_servers()

    # Track aggregated prefixes
    aggregated_prefixes: set[str] = set()

    for server_name in mcp_servers.keys():
        # Check if this is a prefixed server (e.g., "awslabs.xyz")
        if "." in server_name:
            prefix = server_name.split(".")[0]
            if prefix not in aggregated_prefixes:
                aggregated_prefixes.add(prefix)
                # Add aggregated tool with description
                desc = TOOL_DESCRIPTIONS.get(prefix, f"{prefix} services and tools")
                tools[prefix] = desc
        else:
            # Non-prefixed server - add directly if not already present
            if server_name not in tools:
                desc = TOOL_DESCRIPTIONS.get(server_name, f"{server_name} tools")
                tools[server_name] = desc

    return tools


def get_mcp_servers_for_tool(tool_name: str) -> list[str]:
    """
    Get the list of MCP server names that match a tool.

    For aggregated tools like "awslabs", returns all servers starting with "awslabs.".
    For direct tools like "arxiv", returns ["arxiv"].
    """
    mcp_servers = list_mcp_servers()

    # Check if this is an aggregated prefix
    matching_servers = [
        name for name in mcp_servers.keys()
        if name.startswith(f"{tool_name}.")
    ]

    if matching_servers:
        return matching_servers

    # Direct match
    if tool_name in mcp_servers:
        return [tool_name]

    # Handle aliases
    aliases = {"huggingface": "hugging-face", "hugging-face": "huggingface"}
    if tool_name in aliases and aliases[tool_name] in mcp_servers:
        return [aliases[tool_name]]

    return [tool_name]  # Return as-is for built-in tools


_RECENCY_PATTERN = re.compile(
    r"\b(latest|newest|recent|current|new|today|this year|this month|"
    r"state[- ]of[- ]the[- ]art|sota|trending|emerging|just released|"
    r"up[- ]to[- ]date|cutting[- ]edge|most recent|"
    r"\b20\d{2}\b)\b",
    re.IGNORECASE,
)


def _current_date_str() -> str:
    """Return today's date formatted for prompts."""
    return datetime.now().strftime("%B %d, %Y")


def _current_year() -> int:
    return datetime.now().year


def _enrich_query_with_date(query: str) -> str:
    """If *query* implies recency but lacks a year, append the current year."""
    year = str(_current_year())
    if _RECENCY_PATTERN.search(query) and year not in query:
        return f"{query} {year}"
    return query


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
    inferred_entity: str = ""  # What we inferred from ambiguous user input (e.g., typo correction)


# Deliberation prompt template - {available_tools} is filled dynamically
DELIBERATION_PROMPT = """You are deciding how to help a user. Think carefully before acting.

TODAY'S DATE: {current_date}

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
{available_tools}

THINK THROUGH:
1. What does the user ACTUALLY need? (Not just what they literally asked)
2. Is this a follow-up about something already discussed? (Use context if so)
3. Does this require up-to-date information? If YES → action MUST be "use_tools"
4. Can I answer RELIABLY from general knowledge alone? (Only for timeless facts)

MANDATORY TOOL USAGE — you MUST set action to "use_tools" when:
- The query mentions "latest", "recent", "new", "current", "trending", "state-of-the-art", "SOTA", or any year >= 2025
- The query asks about specific papers, repos, models, or datasets (you must verify they exist)
- The query is about pricing, availability, or features of any product/service
- The query asks "what exists" or "what's available" in any research area

WHEN YOU MAY respond_directly (WITHOUT tools):
- Follow-up questions about entities already discussed in this conversation
- Timeless conceptual questions (e.g., "what is backpropagation?")
- Requests to reformat, summarize, or compare things already in context

OTHER RULES:
- For queries about recent/latest topics, ALWAYS include the current year ({current_year}) in your search query
- For AWS-related queries (pricing, services, infrastructure), use awslabs tools
- For news/current events, use web search

INTELLIGENT INTERPRETATION (BE SMART ABOUT USER INTENT):
- When users mention model/product names with typos or variations, INFER their intent:
  * "Haihu" → "Haiku", "cladue" → "claude", "gppt" → "gpt"
  * Accept version variations: "4.5", "4-5", "45" are the same
- For pricing/cost queries about models or services:
  * ALWAYS use awslabs tools to fetch current pricing data - do NOT rely on memorized prices
  * Include your best interpretation of which model/service they mean in the query

CRITICAL - YOUR KNOWLEDGE IS OUTDATED (training cutoff: early 2025):
- Today is {current_date} ({current_year}). Do NOT claim it is 2024 or 2025.
- NEVER say "papers from {current_year} are not yet available" — they ARE available via tools.
- NEVER generate paper titles, arxiv IDs, or repo names from memory — they are likely hallucinated.
- NEVER claim a model/paper/repo "doesn't exist" or "isn't released yet" — SEARCH FIRST.
- When in doubt, ALWAYS search. Let tool results tell you what exists.

Respond with JSON only:
{{
    "understanding": "what user actually needs (including inferred model/service if ambiguous)",
    "inferred_entity": "if user mentioned something ambiguous, what you think they meant",
    "can_answer_from_context": true or false,
    "knowledge_gaps": ["what I don't know that would help"],
    "tools_to_use": [
        {{"tool": "<tool_name>", "rationale": "why this tool", "query": "search query with your best interpretation"}}
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
        web_search_service: ParallelWebSearchService | None = None,
        memory_service: Any | None = None,
    ):
        super().__init__(llm_router, mcp_client, metrics, memory_service=memory_service)
        self.agents = agents or {}
        self._web_search_service: ParallelWebSearchService | None = web_search_service
        self._user_models: dict[str, UserModel] = {}
        self._contexts: dict[str, ConversationContext] = {}

    @property
    def system_prompt(self) -> str:
        return f"""You are DOVA's Thinking Orchestrator, an AI research assistant.

TODAY'S DATE: {_current_date_str()}
YOUR KNOWLEDGE CUTOFF: Early 2025. Anything after that MUST come from tool searches.

Critical rules:
- NEVER generate, guess, or hallucinate paper titles, arxiv IDs, repo names, or model names.
- For ANY query about recent/latest/new/current/trending research, papers, models, or repos:
  you MUST search using tools. NEVER answer from memory.
- The year is {_current_year()}. Do NOT claim it is an earlier year.
- If tools return no results, say so honestly — do NOT fabricate results."""

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
            allowed_sources = task.params.get("sources")
            max_results = task.params.get("max_results")

            deliberation = await self._deliberate(
                query, user_model, context, allowed_sources=allowed_sources,
            )

            # Override: if the query implies recency, force tool usage
            if (
                deliberation.action == ActionDecision.RESPOND_DIRECTLY
                and _RECENCY_PATTERN.search(query)
            ):
                self._logger.info(
                    "overriding_respond_directly_to_use_tools",
                    reason="query implies recency",
                    query=query,
                )
                deliberation.action = ActionDecision.USE_TOOLS
                # If deliberation didn't suggest any tools, add default search tools
                if not any(t.would_help for t in deliberation.tools_to_use):
                    default_tools = []
                    if allowed_sources is None or "arxiv" in (allowed_sources or []):
                        default_tools.append(ToolConsideration(
                            tool_name="arxiv", would_help=True,
                            rationale="Query implies recency — searching arxiv",
                            search_query=query,
                        ))
                    if allowed_sources is None or "web" in (allowed_sources or []):
                        default_tools.append(ToolConsideration(
                            tool_name="web", would_help=True,
                            rationale="Query implies recency — searching web",
                            search_query=query,
                        ))
                    if not default_tools:
                        # Use whatever sources are allowed
                        for src in (allowed_sources or []):
                            if src in ("arxiv", "github", "huggingface", "web"):
                                default_tools.append(ToolConsideration(
                                    tool_name=src, would_help=True,
                                    rationale="Query implies recency — searching " + src,
                                    search_query=query,
                                ))
                                break
                    deliberation.tools_to_use.extend(default_tools)

            # Ensure every user-selected source is represented in tools_to_use.
            # The LLM deliberation decides *which* tools to use, but when the user
            # explicitly selected sources we treat that as a floor, not just a filter.
            if (
                deliberation.action == ActionDecision.USE_TOOLS
                and allowed_sources
            ):
                present_tools = {
                    t.tool_name for t in deliberation.tools_to_use if t.would_help
                }
                # Normalise: treat "hugging-face" as equivalent to "huggingface"
                if "huggingface" in present_tools or "hugging-face" in present_tools:
                    present_tools.update({"huggingface", "hugging-face"})

                for src in allowed_sources:
                    if src not in present_tools:
                        deliberation.tools_to_use.append(
                            ToolConsideration(
                                tool_name=src,
                                would_help=True,
                                rationale=f"User explicitly selected {src}",
                                search_query=query,
                            )
                        )
                        self._logger.info(
                            "added_missing_user_source",
                            source=src,
                            query=query,
                        )

            self._logger.info(
                "deliberation_complete",
                action=deliberation.action.value,
                tools_count=len(deliberation.tools_to_use),
                can_answer_from_context=deliberation.can_answer_from_context,
                allowed_sources=allowed_sources,
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
                tool_results = await self._execute_selected_tools(
                    deliberation, allowed_sources=allowed_sources, max_results=max_results,
                )
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
        allowed_sources: list[str] | None = None,
    ) -> Deliberation:
        """
        Reason about what user needs and decide on action.

        This is the core innovation - explicit reasoning before action.

        Args:
            allowed_sources: If set, only these tools are shown to the LLM
                during deliberation (mirrors user's source toggles).
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

        # Build available tools list -- only include tools the user enabled
        all_tools = get_available_tools()
        if allowed_sources is not None:
            # Normalise alias set so "huggingface" also matches "hugging-face"
            allowed_norm: set[str] = set(allowed_sources)
            for s in list(allowed_norm):
                if s == "huggingface":
                    allowed_norm.add("hugging-face")
                elif s == "hugging-face":
                    allowed_norm.add("huggingface")
            # Always keep non-builtin tools (image, awslabs, etc.) visible
            builtin_names = {"arxiv", "github", "huggingface", "hugging-face", "web"}
            available_tools = {
                name: desc
                for name, desc in all_tools.items()
                if name in allowed_norm or name not in builtin_names
            }
        else:
            available_tools = all_tools

        tools_list = "\n".join(
            f"- {name}: {desc}" for name, desc in available_tools.items()
        )

        # Build and execute deliberation prompt
        prompt = DELIBERATION_PROMPT.format(
            query=query,
            current_date=_current_date_str(),
            current_year=_current_year(),
            expertise_areas=expertise_str,
            communication_style=style_str,
            session_goals=goals_str,
            current_topic=context.current_topic or "none",
            entities_discussed=entities_str,
            recent_turns=recent_turns or "none",
            available_tools=tools_list,
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
                inferred_entity=data.get("inferred_entity", ""),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self._logger.warning("deliberation_parse_error", error=str(e), response=response[:200])
            return Deliberation(
                understanding="Unable to parse deliberation",
                can_answer_from_context=False,
                action=ActionDecision.RESPOND_DIRECTLY,
                reasoning="Parse error, defaulting to direct response",
                inferred_entity="",
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

        prompt = f"""Generate a helpful response based on this context.

TODAY'S DATE: {_current_date_str()}
IMPORTANT: NEVER fabricate paper titles, arxiv IDs, repo names, or model names.
Only reference items that appear in the context below. If you don't have specific
information, say so — do not invent citations.

{chr(10).join(context_parts)}

{style_instructions}

Response:"""

        return await self.think(
            prompt,
            task_type=TaskType.CHAT,
            temperature=0.5,
        )

    async def _execute_selected_tools(
        self,
        deliberation: Deliberation,
        allowed_sources: list[str] | None = None,
        max_results: int | None = None,
    ) -> dict[str, Any]:
        """Execute only the tools that deliberation decided are needed.

        Args:
            deliberation: The deliberation result with tools_to_use.
            allowed_sources: If set, only execute tools whose name is in this list.
                             None means no filtering (all tools allowed).
            max_results: Per-source result limit. None means use defaults.
        """
        # Deduplicate tools_to_use by tool_name, keeping first occurrence.
        # The LLM sometimes suggests the same tool twice with different queries.
        seen_tools: set[str] = set()
        deduped_tools: list[ToolConsideration] = []
        for t in deliberation.tools_to_use:
            norm = "huggingface" if t.tool_name == "hugging-face" else t.tool_name
            if norm not in seen_tools:
                seen_tools.add(norm)
                deduped_tools.append(t)
        deliberation.tools_to_use = deduped_tools

        results: dict[str, Any] = {
            "papers": [],
            "repositories": [],
            "models": [],
            "datasets": [],
            "web_results": [],
            "images": [],
            "mcp_results": [],  # For dynamic MCP tool results
        }

        # Normalise allowed_sources for comparison (e.g. "hugging-face" → "huggingface")
        # Note: None means "no filtering" while [] means "nothing allowed"
        allowed_set: set[str] | None = None
        if allowed_sources is not None:
            allowed_set = set()
            for s in allowed_sources:
                allowed_set.add(s)
                if s == "huggingface":
                    allowed_set.add("hugging-face")
                elif s == "hugging-face":
                    allowed_set.add("huggingface")

        limit = max_results or 10

        for tool in deliberation.tools_to_use:
            if not tool.would_help:
                continue

            # Filter by allowed sources when specified
            # Always allow "image" and dynamic MCP tools through
            if allowed_set is not None and tool.tool_name not in allowed_set:
                builtin_sources = {"arxiv", "github", "huggingface", "hugging-face", "web"}
                if tool.tool_name in builtin_sources:
                    self._logger.info(
                        "tool_skipped_by_source_filter",
                        tool=tool.tool_name,
                        allowed=list(allowed_set),
                    )
                    continue

            try:
                # Enrich query with current year when it implies recency
                search_query = _enrich_query_with_date(tool.search_query)

                self._logger.info(
                    "executing_tool",
                    tool=tool.tool_name,
                    query=search_query,
                    rationale=tool.rationale,
                )

                # Built-in tools
                if tool.tool_name == "arxiv":
                    arxiv_results = await self._search_arxiv(search_query, max_results=limit)
                    results["papers"].extend(arxiv_results)

                elif tool.tool_name == "github":
                    github_results = await self._search_github(search_query, max_results=limit)
                    results["repositories"].extend(github_results)

                elif tool.tool_name in ("huggingface", "hugging-face"):
                    hf_results = await self._search_huggingface(search_query, max_results=limit)
                    results["models"].extend(hf_results.get("models", []))
                    results["datasets"].extend(hf_results.get("datasets", []))

                elif tool.tool_name == "web":
                    web_results = await self._search_web(search_query, max_results=limit)
                    results["web_results"].extend(web_results)

                elif tool.tool_name == "image":
                    enhanced_prompt = await self._enhance_image_prompt(tool.search_query)
                    image_results = await self._generate_image(enhanced_prompt)
                    results["images"].extend(image_results)

                else:
                    # Dynamic MCP tool - could be aggregated (awslabs) or direct
                    mcp_results = await self._execute_mcp_tool(
                        tool.tool_name, search_query
                    )
                    results["mcp_results"].extend(mcp_results)

            except Exception as e:
                self._logger.warning("tool_execution_error", tool=tool.tool_name, error=str(e))

        # Deduplicate results by natural key
        def _dedup(items: list[dict], key: str) -> list[dict]:
            seen: set[str] = set()
            out: list[dict] = []
            for item in items:
                val = item.get(key, "")
                if val and val in seen:
                    continue
                if val:
                    seen.add(val)
                out.append(item)
            return out

        results["papers"] = _dedup(results["papers"], "arxiv_id")
        results["repositories"] = _dedup(results["repositories"], "name")
        results["models"] = _dedup(results["models"], "id")
        results["web_results"] = _dedup(results["web_results"], "url")

        return results

    async def _execute_mcp_tool(self, tool_name: str, query: str) -> list[dict[str, Any]]:
        """
        Execute an MCP tool by name.

        For aggregated tools like "awslabs", selects the best matching server.
        """
        results: list[dict[str, Any]] = []

        # Get matching MCP servers for this tool
        servers = get_mcp_servers_for_tool(tool_name)

        if not servers:
            self._logger.warning("no_mcp_servers_for_tool", tool=tool_name)
            return results

        # Select the best server based on the query
        server_name = await self._select_best_mcp_server(servers, query)

        self._logger.info(
            "executing_mcp_server",
            tool=tool_name,
            server=server_name,
            query=query,
        )

        # Determine the appropriate tool to call on the server
        tool_to_call = self._get_mcp_tool_for_query(server_name, query)

        # Get tool-specific parameters
        tool_params = self._get_mcp_tool_params(server_name, tool_to_call, query)

        try:
            result = await self.call_tool(
                server_name,
                tool_to_call,
                tool_params,
            )

            if result.success and result.data:
                results.append({
                    "source": server_name,
                    "tool": tool_to_call,
                    "data": result.data,
                })
            elif result.error:
                self._logger.warning(
                    "mcp_tool_error",
                    server=server_name,
                    tool=tool_to_call,
                    error=result.error,
                )
        except Exception as e:
            self._logger.warning(
                "mcp_tool_exception",
                server=server_name,
                error=str(e),
            )

        return results

    async def _select_best_mcp_server(self, servers: list[str], query: str) -> str:
        """Select the best MCP server for a query from a list of candidates."""
        if len(servers) == 1:
            return servers[0]

        # Simple keyword matching for AWS servers
        query_lower = query.lower()

        # Priority keywords for AWS server selection
        server_keywords: dict[str, list[str]] = {
            "awslabs.aws-pricing-mcp-server": ["pricing", "cost", "price", "estimate", "budget"],
            "awslabs.aws-documentation-mcp-server": ["documentation", "docs", "guide", "how to", "best practice"],
            "awslabs.bedrock-kb-retrieval-mcp-server": ["knowledge base", "bedrock kb", "rag", "retrieval"],
            "awslabs.cdk-mcp-server": ["cdk", "infrastructure as code", "iac"],
            "awslabs.cloudwatch-mcp-server": ["cloudwatch", "logs", "metrics", "monitoring", "alarm"],
            "awslabs.lambda-tool-mcp-server": ["lambda", "serverless", "function"],
            "awslabs.eks-mcp-server": ["eks", "kubernetes", "k8s", "container"],
            "awslabs.dynamodb-mcp-server": ["dynamodb", "nosql", "table"],
            "awslabs.s3-tables-mcp-server": ["s3", "storage", "bucket"],
            "awslabs.iam-mcp-server": ["iam", "permission", "role", "policy", "access"],
            "awslabs.cfn-mcp-server": ["cloudformation", "cfn", "stack", "template"],
            "awslabs.terraform-mcp-server": ["terraform", "tf", "hcl"],
        }

        # Score each server
        best_server = servers[0]
        best_score = 0

        for server in servers:
            keywords = server_keywords.get(server, [])
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > best_score:
                best_score = score
                best_server = server

        # Default to docs server for general AWS queries if no specific match
        if best_score == 0 and "awslabs.aws-documentation-mcp-server" in servers:
            return "awslabs.aws-documentation-mcp-server"

        return best_server

    def _get_mcp_tool_for_query(self, server_name: str, query: str) -> str:
        """Get the appropriate tool name for an MCP server."""
        # Map server names to their primary search/query tools
        tool_mapping: dict[str, str] = {
            "awslabs.aws-documentation-mcp-server": "search_documentation",
            "awslabs.aws-pricing-mcp-server": "get_pricing",
            "awslabs.bedrock-kb-retrieval-mcp-server": "QueryKnowledgeBases",
            "awslabs.cdk-mcp-server": "CDKGeneralGuidance",
            "awslabs.cloudwatch-mcp-server": "describe_log_groups",
            "awslabs.cfn-mcp-server": "get_resource_schema_information",
            "awslabs.terraform-mcp-server": "SearchAwsProviderDocs",
            "awslabs.eks-mcp-server": "list_k8s_resources",
            "awslabs.lambda-tool-mcp-server": "fastapi_server",
            "awslabs.dynamodb-mcp-server": "dynamodb_data_modeling",
            "awslabs.s3-tables-mcp-server": "list_tables",
            "awslabs.iam-mcp-server": "list_roles",
        }

        return tool_mapping.get(server_name, "search")

    def _get_mcp_tool_params(
        self, server_name: str, tool_name: str, query: str
    ) -> dict[str, Any]:
        """Get tool-specific parameters based on server and tool name."""
        # Extract service/model info from query for pricing
        query_lower = query.lower()

        # AWS Documentation - uses search_phrase
        if server_name == "awslabs.aws-documentation-mcp-server":
            return {"search_phrase": query}

        # AWS Pricing - needs service_code and optional filters
        if server_name == "awslabs.aws-pricing-mcp-server":
            # Extract keywords from query for later fuzzy matching
            # Store them in context for the synthesis phase
            keywords = self._extract_model_keywords(query_lower)
            self._logger.debug("extracted_model_keywords", keywords=keywords, query=query_lower)

            # Determine service based on query content
            # Don't pre-filter - let AWS return available options, then fuzzy match
            if "ec2" in query_lower and "bedrock" not in query_lower:
                return {"service_code": "AmazonEC2", "region": "us-east-1"}
            elif "s3" in query_lower and "bedrock" not in query_lower:
                return {"service_code": "AmazonS3", "region": "us-east-1"}
            elif "lambda" in query_lower and "bedrock" not in query_lower:
                return {"service_code": "AWSLambda", "region": "us-east-1"}
            elif "dynamodb" in query_lower and "bedrock" not in query_lower:
                return {"service_code": "AmazonDynamoDB", "region": "us-east-1"}
            else:
                # Default to Bedrock for model queries - let AWS return all models
                # The synthesis phase will fuzzy-match to find the best one
                return {
                    "service_code": "AmazonBedrock",
                    "region": "us-east-1",
                    "_query_keywords": keywords,  # Pass keywords for synthesis phase
                }

        # Bedrock Knowledge Base - needs query
        if server_name == "awslabs.bedrock-kb-retrieval-mcp-server":
            return {"query": query}

        # CDK Guidance - uses question
        if server_name == "awslabs.cdk-mcp-server":
            return {"question": query}

        # Terraform docs - uses query
        if server_name == "awslabs.terraform-mcp-server":
            return {"query": query}

        # Default fallback - try common parameter names
        return {"query": query}

    def _extract_model_keywords(self, query_lower: str) -> list[str]:
        """
        Extract potential model-related keywords from user query for fuzzy matching.

        Instead of hardcoding model names, extract keywords that can be used
        to search/filter against dynamically fetched model lists.
        Returns list of keywords to use for matching.
        """
        import re

        # Extract all words that might be model identifiers
        # Look for patterns like: version numbers, model family names, sizes
        keywords = []

        # Extract words (alphanumeric sequences)
        words = re.findall(r'[a-z0-9]+(?:[.-][a-z0-9]+)*', query_lower)

        # Filter to words that are likely model-related (not common stop words)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'just', 'and', 'but', 'if', 'or', 'because', 'until',
            'while', 'what', 'which', 'who', 'whom', 'this', 'that',
            'these', 'those', 'am', 'price', 'pricing', 'cost', 'estimate',
            'create', 'using', 'use', 'model', 'models', 'bedrock', 'amazon', 'aws',
            'tokens', 'token', 'input', 'output', 'rpm', 'request', 'requests',
            'per', 'minute', 'monthly', 'daily', 'average', 'avg',
        }

        for word in words:
            if word not in stop_words and len(word) > 1:
                keywords.append(word)

        # Also look for version patterns like "4.5", "3.5", "120b", "70b"
        version_patterns = re.findall(r'\d+\.?\d*[bkm]?', query_lower)
        keywords.extend(version_patterns)

        return keywords

    def _fuzzy_match_score(self, query_keywords: list[str], model_id: str) -> float:
        """
        Calculate fuzzy match score between query keywords and a model ID.

        Higher score = better match. Uses simple substring matching and
        Levenshtein-like similarity without external dependencies.
        """
        if not query_keywords:
            return 0.0

        model_lower = model_id.lower()
        score = 0.0

        for keyword in query_keywords:
            # Exact substring match
            if keyword in model_lower:
                score += 2.0
            else:
                # Check for partial/fuzzy match (simple character overlap)
                overlap = sum(1 for c in keyword if c in model_lower)
                if overlap > len(keyword) * 0.6:  # 60% character overlap
                    score += 0.5

        # Normalize by number of keywords
        return score / len(query_keywords) if query_keywords else 0.0

    def _process_pricing_with_fuzzy_match(
        self, query: str, pricing_data: dict | list
    ) -> str | None:
        """
        Process pricing data with fuzzy matching to find the best model match.

        This enables intelligent model identification even with typos or variations.
        Returns formatted pricing info for the best matching model(s).
        """
        query_lower = query.lower()
        keywords = self._extract_model_keywords(query_lower)

        if not keywords:
            return None

        self._logger.debug("fuzzy_matching_pricing", keywords=keywords)

        # Extract model entries from pricing data
        models_with_pricing = []

        if isinstance(pricing_data, list):
            for item in pricing_data:
                if isinstance(item, dict):
                    # Try common fields for model identification
                    model_id = (
                        item.get("modelId") or
                        item.get("model_id") or
                        item.get("productFamily") or
                        item.get("usagetype") or
                        str(item)
                    )
                    models_with_pricing.append((model_id, item))
        elif isinstance(pricing_data, dict):
            # Check for nested price list or models
            if "PriceList" in pricing_data:
                for price_item in pricing_data["PriceList"]:
                    if isinstance(price_item, str):
                        try:
                            price_item = json.loads(price_item)
                        except json.JSONDecodeError:
                            continue
                    if isinstance(price_item, dict):
                        product = price_item.get("product", {})
                        attrs = product.get("attributes", {})
                        model_id = attrs.get("modelId", attrs.get("usagetype", "unknown"))
                        models_with_pricing.append((model_id, price_item))
            else:
                # Single item
                model_id = pricing_data.get("modelId", str(pricing_data)[:100])
                models_with_pricing.append((model_id, pricing_data))

        if not models_with_pricing:
            return None

        # Score each model against the query keywords
        scored_models = []
        for model_id, data in models_with_pricing:
            score = self._fuzzy_match_score(keywords, str(model_id))
            if score > 0:
                scored_models.append((score, model_id, data))

        if not scored_models:
            # No good matches - return all available for LLM to interpret
            return f"**Available Models (no exact match for '{' '.join(keywords)}'):**\n" + \
                   "\n".join(f"- {m[0]}" for m in models_with_pricing[:10])

        # Sort by score descending
        scored_models.sort(reverse=True, key=lambda x: x[0])

        # Get best match and confidence
        best_score, best_model_id, best_data = scored_models[0]
        confidence = "high" if best_score > 1.5 else "medium" if best_score > 0.8 else "low"

        # Format the result
        result_lines = [
            f"**Best Match ({confidence} confidence): {best_model_id}**",
            f"(Matched keywords: {', '.join(keywords)})",
        ]

        # Extract pricing info if available
        if isinstance(best_data, dict):
            if "terms" in best_data:
                # AWS pricing format
                on_demand = best_data.get("terms", {}).get("OnDemand", {})
                for term_key, term_data in on_demand.items():
                    price_dims = term_data.get("priceDimensions", {})
                    for dim_key, dim_data in price_dims.items():
                        price = dim_data.get("pricePerUnit", {}).get("USD", "N/A")
                        unit = dim_data.get("unit", "")
                        desc = dim_data.get("description", "")[:100]
                        result_lines.append(f"  - ${price} per {unit}: {desc}")
            else:
                # Generic format
                result_lines.append(f"  Data: {json.dumps(best_data, indent=2)[:300]}")

        # Show alternatives if close matches exist
        if len(scored_models) > 1:
            alternatives = [m[1] for m in scored_models[1:4] if m[0] > 0.5]
            if alternatives:
                result_lines.append(f"\n**Similar models:** {', '.join(str(a) for a in alternatives)}")

        return "\n".join(result_lines)

    async def _search_arxiv(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """Search ArXiv for papers."""
        result = await self.search_arxiv(query, max_results=max_results)
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

    async def _search_github(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """Search GitHub for repositories."""
        result = await self.search_github(query, search_type="repositories", per_page=max_results)
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

    async def _search_huggingface(self, query: str, max_results: int = 10) -> dict[str, list[dict[str, Any]]]:
        """Search HuggingFace for models and datasets."""
        results: dict[str, list[dict[str, Any]]] = {"models": [], "datasets": []}

        # Search models
        model_result = await self.search_huggingface(query, search_type="models", limit=max_results)
        if model_result.success and model_result.data:
            models = model_result.data if isinstance(model_result.data, list) else [model_result.data]
            results["models"] = [
                {
                    "id": m.get("id", m.get("modelId", "")),
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                    "tags": m.get("tags", []),
                    "pipeline_tag": m.get("pipeline_tag", ""),
                    "library_name": m.get("library_name", ""),
                    "last_modified": m.get("lastModified", m.get("last_modified", "")),
                }
                for m in models if isinstance(m, dict)
            ]

        # Search datasets
        dataset_result = await self.search_huggingface(query, search_type="datasets", limit=max(max_results // 2, 3))
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

    async def _search_web(self, query: str, max_results: int = 15) -> list[dict[str, Any]]:
        """Search web using parallel multi-provider service with aggregation."""
        if self._web_search_service is None:
            self._web_search_service = create_parallel_search_service()

        try:
            # Use parallel search across all available providers
            response = await self._web_search_service.search_parallel(
                query=query,
                max_results=max_results,
            )

            if response.error:
                self._logger.warning(
                    "parallel_web_search_error",
                    error=response.error,
                    providers_failed=response.providers_failed,
                )
                return []

            self._logger.info(
                "parallel_web_search_complete",
                query=query,
                providers_succeeded=response.providers_succeeded,
                providers_failed=list(response.providers_failed.keys()),
                result_count=len(response.results),
                deduplicated=response.deduplicated_count,
                time_ms=response.total_time_ms,
            )

            return [
                {
                    "title": r.title,
                    "url": r.url,
                    "description": r.snippet[:500] if r.snippet else "",
                    "published_date": r.published_date,
                    "source_provider": r.source_provider,
                }
                for r in response.results
            ]
        except Exception as e:
            self._logger.warning("web_search_error", error=str(e))
            return []

    async def _enhance_image_prompt(self, user_prompt: str) -> str:
        """Enhance user prompt for better image generation."""
        enhancement_prompt = f"""Rephrase this image request into an optimal prompt for AI image generation.

User request: {user_prompt}

Create a detailed, descriptive prompt that:
1. Describes the main subject clearly
2. Includes style descriptors (e.g., photorealistic, digital art, watercolor)
3. Specifies lighting, mood, and atmosphere
4. Adds quality keywords (high quality, detailed, 4k)

Return ONLY the enhanced prompt, nothing else."""

        return await self.think(enhancement_prompt, temperature=0.7)

    async def _generate_image(self, prompt: str) -> list[dict[str, Any]]:
        """Generate image using HuggingFace Z-Image-Turbo."""
        self._logger.info("generating_image", prompt=prompt[:100])

        result = await self.generate_image(prompt)

        if not result.success:
            self._logger.warning("image_generation_failed", error=result.error)
            return []

        # Parse result - the MCP tool returns (gallery_images, seed_str, seed_int)
        data = result.data
        if isinstance(data, (list, tuple)) and len(data) >= 1:
            # gallery_images is the first element, could be a list of image URLs/paths
            gallery = data[0] if isinstance(data[0], list) else [data[0]]
            seed = data[2] if len(data) > 2 else 0
            return [
                {
                    "url": img.get("url", img) if isinstance(img, dict) else str(img),
                    "prompt": prompt,
                    "resolution": "1024x1024",
                    "seed": seed,
                }
                for img in gallery
                if img
            ]
        elif isinstance(data, dict):
            return [{
                "url": data.get("url", ""),
                "prompt": prompt,
                "resolution": "1024x1024",
                "seed": data.get("seed", 0),
            }]

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
            paper_lines = []
            for p in papers:
                title = p.get('title', 'Unknown')
                url = p.get('url', '')
                authors = p.get('metadata', {}).get('authors', p.get('authors', []))
                author_str = ', '.join(authors[:3]) if authors else ''
                arxiv_id = p.get('metadata', {}).get('arxiv_id', p.get('arxiv_id', ''))
                abstract = p.get('description', p.get('summary', ''))[:300]
                line = f"- **{title}**"
                if arxiv_id:
                    line += f"\n  ArXiv: {arxiv_id}"
                if url:
                    line += f"\n  URL: {url}"
                if author_str:
                    line += f"\n  Authors: {author_str}"
                if abstract:
                    line += f"\n  Abstract: {abstract}"
                paper_lines.append(line)
            result_parts.append(f"**Papers Found ({len(results['papers'])}):**\n" + "\n".join(paper_lines))

        if results.get("repositories"):
            repos = results["repositories"][:5]
            repo_lines = []
            for r in repos:
                name = r.get('name', r.get('full_name', 'Unknown'))
                desc = r.get('description', '')[:150]
                url = r.get('url', r.get('html_url', ''))
                stars = r.get('metadata', {}).get('stars', r.get('stars', 0))
                lang = r.get('metadata', {}).get('language', r.get('language', ''))
                line = f"- **{name}** ({stars} stars, {lang}): {desc}"
                if url:
                    line += f"\n  URL: {url}"
                repo_lines.append(line)
            result_parts.append(f"**Repositories Found ({len(results['repositories'])}):**\n" + "\n".join(repo_lines))

        if results.get("models"):
            models = results["models"][:5]
            model_lines = []
            for m in models:
                mid = m.get('id', m.get('title', 'Unknown'))
                downloads = m.get('metadata', {}).get('downloads', m.get('downloads', 0))
                tags = m.get('metadata', {}).get('tags', m.get('tags', []))
                pipeline = m.get('metadata', {}).get('pipeline_tag', m.get('pipeline_tag', ''))
                library = m.get('library_name', '')
                tag_str = ', '.join(tags[:5]) if tags else ''
                url = m.get('url', f'https://huggingface.co/{mid}')
                line = f"- **{mid}** ({downloads} downloads, {pipeline})"
                if library:
                    line += f"\n  Library: {library}"
                if tag_str:
                    line += f"\n  Tags: {tag_str}"
                if url:
                    line += f"\n  URL: {url}"
                model_lines.append(line)
            result_parts.append(f"**Models Found ({len(results['models'])}):**\n" + "\n".join(model_lines))

        if results.get("web_results"):
            web = results["web_results"][:5]
            web_lines = []
            for w in web:
                title = w.get('title', 'Unknown')
                desc = w.get('description', '')[:200]
                url = w.get('url', '')
                line = f"- **{title}**: {desc}"
                if url:
                    line += f"\n  URL: {url}"
                web_lines.append(line)
            result_parts.append(f"**Web Results ({len(results['web_results'])}):**\n" + "\n".join(web_lines))

        if results.get("images"):
            images = results["images"]
            result_parts.append(f"**Images Generated ({len(images)}):** Images have been created based on your request.")

        if results.get("mcp_results"):
            for mcp_result in results["mcp_results"]:
                source = mcp_result.get("source", "unknown")
                data = mcp_result.get("data", {})

                # Special handling for pricing results - apply fuzzy matching
                if "pricing" in source.lower() and isinstance(data, (dict, list)):
                    fuzzy_result = self._process_pricing_with_fuzzy_match(query, data)
                    if fuzzy_result:
                        result_parts.append(fuzzy_result)
                        continue

                # Format based on data type
                if isinstance(data, dict):
                    summary = json.dumps(data, indent=2)[:500]
                elif isinstance(data, list):
                    summary = f"{len(data)} items returned"
                else:
                    summary = str(data)[:500]
                result_parts.append(f"**{source}:**\n{summary}")

        # Get personalization
        style_instructions = self._get_style_instructions(user_model)

        # Add inferred entity context if available
        inferred_context = ""
        if hasattr(deliberation, 'inferred_entity') and deliberation.inferred_entity:
            inferred_context = f"\nNote: User likely meant '{deliberation.inferred_entity}' - adjust response accordingly.\n"

        prompt = f"""Synthesize these search results into a helpful response.

TODAY'S DATE: {_current_date_str()}

User Question: {query}
Understanding: {deliberation.understanding}
{inferred_context}
Search Results:
{chr(10).join(result_parts) if result_parts else "No results found."}

{style_instructions}

CRITICAL RULES - FOLLOW EXACTLY:

1. YOUR KNOWLEDGE IS STALE - TRUST TOOL RESULTS
   - FORBIDDEN PHRASES: "doesn't exist", "not released yet", "as of my last update",
     "my knowledge cutoff", "not yet available", "we're currently in [year]"
   - Today is {_current_date_str()}. NEVER claim it is an earlier year.
   - If tool returned data, USE IT. If tool returned no data, say "no results found for this query"
   - NEVER fabricate paper titles, arxiv IDs, repo names, or model names that are not in the search results above
   - ONLY reference items that actually appear in the Search Results section

2. CORRECT RESPONSE PATTERN FOR PARTIAL MATCHES:
   BAD: "Claude 4.5 Haiku doesn't exist. Here's Claude 3.5 Haiku instead..."
   GOOD: "Here's the pricing for Claude Haiku. Based on the results, the closest match is [model]. [provide pricing]"

3. TYPO HANDLING
   - Fix typos silently or with brief note: "(interpreting 'Haihu' as Haiku)"
   - Don't lecture about the typo

4. PRICING QUERIES
   - Use ONLY data from tool results
   - If exact model not found: extrapolate from closest match, state assumption
   - Show your calculation

5. ALWAYS ANSWER
   - No refusals. Best estimate with confidence level.
   - "Based on similar model pricing, I estimate..." is always better than "I can't help"

6. COMPREHENSIVE OUTPUT
   - Provide in-depth explanations with detailed analysis
   - Include mathematical formulas or equations where relevant (use LaTeX notation: $formula$)
   - ALWAYS include source URLs as inline links for every paper, repo, model, or web source you reference
   - Structure the response with clear sections and headers when covering multiple aspects
   - For technical topics, explain key concepts, methods, and their significance
   - For each key finding or recommendation, explain the RATIONALE: why it matters, what problem it solves, or what makes it significant
   - When comparing approaches, explain trade-offs and under what conditions each excels
   - Connect findings to practical implications — who benefits and how
   - When showing implementation frameworks or algorithmic approaches, use PDL (Program Description Language) style pseudo code instead of Python or any specific programming language. Example PDL format:
     ```
     PROCEDURE TrainModel(data, config)
       preprocessed ← Preprocess(data, config.tokenizer)
       FOR each epoch IN 1..config.num_epochs DO
         loss ← ComputeLoss(model, preprocessed)
         UPDATE model.parameters USING Backprop(loss)
       END FOR
       RETURN model
     END PROCEDURE
     ```
   - PDL pseudo code should be language-agnostic, using clear keywords like PROCEDURE, FOR, IF/THEN/ELSE, WHILE, RETURN, CALL, INPUT, OUTPUT

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
            instructions.append("When showing implementation details, use PDL-style pseudo code (not Python or any specific language).")

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

    @staticmethod
    def extract_research_data(result_data: dict) -> dict:
        """Map orchestrator output to flat research arrays for ResearchResponse."""
        action_result = result_data.get("action_result") or {}
        return {
            "response": result_data.get("response", ""),
            "papers": action_result.get("papers", []),
            "repositories": action_result.get("repositories", []),
            "models": action_result.get("models", []),
            "datasets": action_result.get("datasets", []),
            "web_results": action_result.get("web_results", []),
            "images": action_result.get("images", []),
            "mcp_results": action_result.get("mcp_results", []),
            "deliberation": result_data.get("deliberation", {}),
        }

    def register_agent(self, agent_type: str, agent: BaseAgent) -> None:
        """Register a specialized agent."""
        self.agents[agent_type] = agent
        self._logger.info("agent_registered", agent_type=agent_type)
