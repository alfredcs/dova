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

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable

import structlog

from dova.agents.base import AgentResult, AgentTask, BaseAgent
from dova.agents.conversation_context import ConversationContext, ConversationTurn
from dova.agents.user_model import ExpertiseLevel, ResponseDepth, UserModel
from dova.config.mcp_servers import BIO_MCP_SERVERS, list_mcp_servers
from dova.config.providers import LLMRouter, TaskType
from dova.services.web_search import (
    ParallelWebSearchService,
    create_parallel_search_service,
)
from dova.utils.metrics import MetricsCollector

logger = structlog.get_logger(__name__)

# Progress callback type: (event_type, data) -> awaitable
ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None]] | None


# Tool descriptions for known tool types
TOOL_DESCRIPTIONS: dict[str, str] = {
    "arxiv": "Academic papers (use for: research papers, scientific studies, technical methods)",
    "github": "Code repositories (use for: implementations, libraries, code examples)",
    "huggingface": "ML models/datasets (use for: pretrained models, datasets, ML-specific)",
    "hugging-face": "ML models/datasets (use for: pretrained models, datasets, ML-specific)",
    "web": "Web search (use for: news, current events, general information, non-technical topics)",
    "image": "Image generation (use for: creating images, visualizations, artwork, illustrations)",
    "awslabs": "AWS services (use for: AWS pricing, documentation, CDK, CloudFormation, Bedrock, etc.)",
    "bio": "Biomedical / pharma data (use for: PubMed literature, clinical trials, drug/chemical compounds, gene/protein/disease queries)",
}


def get_available_tools() -> dict[str, str]:
    """
    Load available tools from ~/.dova.json and aggregate prefixed tools.

    Tools with names like "awslabs.xyz" are aggregated into "awslabs".
    Returns a dict of tool_name -> description.
    """
    tools: dict[str, str] = {}

    # Always include built-in tools (bio is always available via hosted HTTP
    # endpoints registered in get_default_registry; no user config needed).
    for name in ["arxiv", "github", "huggingface", "web", "image", "bio"]:
        tools[name] = TOOL_DESCRIPTIONS.get(name, f"{name} search")

    # Load MCP servers from config
    mcp_servers = list_mcp_servers()

    # Track aggregated prefixes
    aggregated_prefixes: set[str] = set()

    # Bio servers are aggregated under the "bio" umbrella — don't surface
    # their individual names (pubmed-bio, pubchem-bio, clinicaltrials-bio)
    # to the deliberation LLM; the orchestrator picks among them by keyword.
    bio_server_set: set[str] = set(BIO_MCP_SERVERS)

    for server_name in mcp_servers.keys():
        # Check if this is a prefixed server (e.g., "awslabs.xyz")
        if "." in server_name:
            prefix = server_name.split(".")[0]
            if prefix not in aggregated_prefixes:
                aggregated_prefixes.add(prefix)
                # Add aggregated tool with description
                desc = TOOL_DESCRIPTIONS.get(prefix, f"{prefix} services and tools")
                tools[prefix] = desc
        elif server_name in bio_server_set:
            # Already represented by the "bio" umbrella above.
            continue
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
    For the "bio" umbrella, returns the curated bio server list.
    For direct tools like "arxiv", returns ["arxiv"].
    """
    # Bio umbrella — always resolves to the curated list; these servers
    # are registered by get_default_registry() and may not appear in
    # list_mcp_servers() (which only reads ~/.dova.json).
    if tool_name == "bio":
        return list(BIO_MCP_SERVERS)

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


_EVALUATIVE_PATTERNS = [
    r"\bevaluate\b", r"\bcompare\b", r"\bvs\.?\b", r"\bversus\b",
    r"\btradeoffs?\b", r"\btrade-offs?\b", r"\bpros?\s+(?:and\s+)?cons?\b",
    r"\badvantages?\s+(?:and\s+)?disadvantages?\b",
    r"\bstrengths?\s+(?:and\s+)?weaknesses?\b",
    r"\bshould\s+(?:i|we)\s+(?:use|choose|pick|adopt)\b",
    r"\bwhich\s+(?:is|are)\s+(?:better|best)\b",
    r"\bdebate\b", r"\barguments?\s+(?:for|against)\b",
]


def _is_evaluative_query(query: str) -> bool:
    """Whether the query asks for an evaluative/debate-style analysis."""
    q = query.lower()
    return any(re.search(p, q) for p in _EVALUATIVE_PATTERNS)


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
    # Semantic intent weights across the three top-level groups. Sum to 1.0.
    # Used downstream to weight result aggregation in synthesis, not execution.
    intent_weights: dict[str, float] = field(default_factory=dict)


# --- Intent scoring vocabulary -------------------------------------------
# These compact keyword lists score how strongly a query leans toward the
# AI / Bio / Web groups. The scorer is intentionally lightweight (no LLM
# round-trip) so it runs cheaply on every query. The bio list is aligned
# with _select_bio_servers so routing and weighting stay consistent.

_AI_INTENT_KEYWORDS: tuple[str, ...] = (
    "neural network", "transformer", "llm", "large language model",
    "reinforcement learning", "rlhf", "dpo", "ppo", "fine-tuning",
    "pretraining", "pre-training", "distillation", "lora", "qlora",
    "attention", "embedding", "tokenizer", "moe", "mixture-of-experts",
    "diffusion model", "gan", "vae", "autoencoder", "foundation model",
    "cnn", "rnn", "lstm", "gradient", "backprop",
    "huggingface", "pytorch", "tensorflow", "jax",
    "arxiv", "benchmark", "sota", "state-of-the-art",
    "agent", "orchestration", "tool use", "tool-use", "react loop",
    "inference", "quantization", "speculative decoding", "kv cache",
    "scaling law", "chinchilla",
    "algorithm", "architecture", "implementation", "open-source",
    "repository", "repo ", "python", "codebase",
)

_BIO_INTENT_KEYWORDS: tuple[str, ...] = (
    # Clinical trials
    "clinical trial", "trial", "nct", "phase i", "phase ii", "phase iii",
    "phase iv", "randomized", "placebo", "enrollment", "recruiting",
    "eligibility", "ind-enabling", "primary endpoint",
    # Compounds / pharmacology
    "compound", "molecule", "small molecule", "smiles", "inchi", "pubchem",
    "cid", "bioassay", "admet", "herg", "cyp", "logp",
    "pharmacokinetic", "pharmacodynamic", "drug-drug interaction", "ddi",
    "hepatotoxicity", "retrosynthesis",
    # Biomedical / genomics / protein
    "protein", "peptide", "antibody", "antigen", "epitope", "binder",
    "binding affinity", "rfdiffusion", "alphafold", "af2", "af3",
    "docking", "cryptic site", "enzyme", "directed evolution",
    "gene", "genome", "genomic", "gwas", "variant", "mutation", "allele",
    "transcriptom", "proteomic", "metabolomic", "single-cell", "scrna",
    "crispr", "gene editing", "base editing", "prime editing", "aav", "lnp",
    "cancer", "oncolog", "tumor", "tumour", "melanoma", "leukemia",
    "lymphoma", "diabetes", "alzheimer", "parkinson", "hepatitis",
    "hiv", "sars-cov", "covid",
    "pubmed", "pmid", "mesh", "medline", "biomedical", "biotech",
    "pharma", "pharmaceutical", "biomarker", "therapeutic",
    "histopathology", "radiology", "ehr", "real-world evidence",
)

_WEB_INTENT_KEYWORDS: tuple[str, ...] = (
    "news", "announced", "nominated", "election", "stock", "earnings",
    "released", "launched", "today", "this week", "this month",
    "price", "pricing", "cost", "market", "regulation", "policy",
    "country", "company", "ceo", "founder", "startup",
    "blog", "twitter", "reddit", "wikipedia",
)


def compute_intent_weights(
    query: str,
    allowed_groups: set[str] | None = None,
    web_floor: float = 0.10,
    group_floor: float = 0.05,
) -> dict[str, float]:
    """
    Score a query's intent distribution across {ai, bio, web}.

    Returns weights in [0, 1] that sum to 1.0. The scorer is keyword-based
    and intentionally simple — its output is used only to weight result
    aggregation during synthesis, not execution. Every allowed group
    receives at least `group_floor`, and web receives at least `web_floor`
    when it's allowed (because general-purpose context usually helps).

    Args:
        query: User query string.
        allowed_groups: Subset of {"ai","bio","web"} the UI selected.
                        None = all three allowed.
        web_floor: Minimum weight for web when allowed.
        group_floor: Minimum weight for any other allowed group that has
                     zero keyword hits (prevents zero-sum exclusion).
    """
    q = query.lower()
    if allowed_groups is None:
        allowed_groups = {"ai", "bio", "web"}

    def _count(kws: tuple[str, ...]) -> int:
        return sum(1 for kw in kws if kw in q)

    raw: dict[str, float] = {}
    if "ai" in allowed_groups:
        raw["ai"] = float(_count(_AI_INTENT_KEYWORDS))
    if "bio" in allowed_groups:
        raw["bio"] = float(_count(_BIO_INTENT_KEYWORDS))
    if "web" in allowed_groups:
        raw["web"] = float(_count(_WEB_INTENT_KEYWORDS))

    # Normalise raw counts into a distribution. If everything is zero
    # (e.g., a very generic query), split evenly across allowed groups.
    total = sum(raw.values())
    if total <= 0:
        share = 1.0 / max(len(raw), 1)
        weights = {g: share for g in raw}
    else:
        weights = {g: v / total for g, v in raw.items()}

    # Enforce floors by redistributing excess weight from over-funded groups.
    # After this block, min weights hold exactly and the distribution still
    # sums to 1.0 — no post-hoc normalisation needed.
    def _floors() -> dict[str, float]:
        floors: dict[str, float] = {}
        for g in weights:
            floors[g] = web_floor if g == "web" else group_floor
        # If floors can't all be satisfied (e.g., a single allowed group),
        # scale the floors down proportionally so they still sum to <= 1.
        f_total = sum(floors.values())
        if f_total > 1.0:
            scale = 1.0 / f_total
            floors = {g: v * scale for g, v in floors.items()}
        return floors

    floors = _floors()
    # Amount we need to "raise" each under-floor group to its floor.
    deficits = {g: max(0.0, floors[g] - weights[g]) for g in weights}
    total_deficit = sum(deficits.values())
    if total_deficit > 0:
        # Pool of "excess" above-floor weight we can donate from.
        excesses = {
            g: max(0.0, weights[g] - floors[g]) for g in weights
        }
        excess_total = sum(excesses.values())
        if excess_total > 0:
            # Each over-floor group contributes proportionally to its excess.
            for g in weights:
                donation = excesses[g] / excess_total * total_deficit
                weights[g] -= donation
            for g in weights:
                weights[g] += deficits[g]

    # Round for log/UI readability — keeps 2 decimals and still sums to 1.0.
    rounded = {g: round(v, 2) for g, v in weights.items()}
    drift = round(1.0 - sum(rounded.values()), 2)
    if rounded and abs(drift) >= 0.01:
        # Put the rounding residue on the largest group.
        top = max(rounded, key=rounded.get)
        rounded[top] = round(rounded[top] + drift, 2)
    return rounded


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

    async def execute(self, task: AgentTask, progress: ProgressCallback = None) -> AgentResult:
        """
        Execute with deliberation-first approach.

        Args:
            task: Task containing the user query
            progress: Optional async callback for streaming progress events.

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

            # 1. Load user model and conversation context in parallel
            user_model, context = await asyncio.gather(
                self._load_user_model(task.user_id),
                self._load_conversation_context(session_id, task.user_id),
            )

            # Add user query to context
            context.add_turn(role="user", content=query)

            # 2. DELIBERATE - the key innovation
            allowed_sources = task.params.get("sources")
            max_results = task.params.get("max_results")

            # Fast path: skip LLM deliberation when user explicitly selected sources
            if allowed_sources and len(allowed_sources) > 0:
                enriched_query = _enrich_query_with_date(query)
                deliberation = Deliberation(
                    understanding=query,
                    can_answer_from_context=False,
                    action=ActionDecision.USE_TOOLS,
                    reasoning="User explicitly selected sources — skipping deliberation",
                    tools_to_use=[
                        ToolConsideration(
                            tool_name=s,
                            would_help=True,
                            rationale=f"User selected {s}",
                            search_query=enriched_query,
                        )
                        for s in allowed_sources
                    ],
                )
                self._logger.info(
                    "fast_path_deliberation",
                    sources=allowed_sources,
                    query=query,
                )
            else:
                deliberation = await self._deliberate(
                    query, user_model, context, allowed_sources=allowed_sources,
                )

            # Compute semantic intent weights across {ai, bio, web}. Used in
            # synthesis for proportional aggregation — not for execution.
            # Map the user's source list to the 3 top-level groups.
            allowed_groups: set[str] | None = None
            if allowed_sources is not None:
                allowed_groups = set()
                ai_sources = {"arxiv", "github", "huggingface", "hugging-face"}
                for s in allowed_sources:
                    if s in ai_sources:
                        allowed_groups.add("ai")
                    elif s == "web":
                        allowed_groups.add("web")
                    elif s == "bio":
                        allowed_groups.add("bio")
            deliberation.intent_weights = compute_intent_weights(
                query, allowed_groups=allowed_groups,
            )
            self._logger.info(
                "intent_weights_computed",
                weights=deliberation.intent_weights,
                allowed_groups=sorted(allowed_groups) if allowed_groups else None,
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

            if progress:
                tools_planned = [t.tool_name for t in deliberation.tools_to_use if t.would_help]
                await progress("stage", {
                    "stage": "deliberating",
                    "message": f"Decided to {'search ' + ', '.join(tools_planned) if deliberation.action == ActionDecision.USE_TOOLS else 'respond directly'}",
                    "action": deliberation.action.value,
                    "tools_planned": tools_planned,
                    "intent_weights": deliberation.intent_weights,
                })
                if deliberation.intent_weights:
                    weights_text = ", ".join(
                        f"{int(v * 100)}% {k.upper()}"
                        for k, v in deliberation.intent_weights.items()
                        if v > 0
                    )
                    await progress("thinking", {
                        "step_type": "deliberation",
                        "content": f"Semantic intent: {weights_text}",
                    })
                await progress("thinking", {
                    "step_type": "observation",
                    "content": deliberation.understanding or query,
                })
                if deliberation.reasoning:
                    await progress("thinking", {
                        "step_type": "reasoning",
                        "content": deliberation.reasoning,
                    })
                await progress("thinking", {
                    "step_type": "plan",
                    "content": (
                        f"Action: {deliberation.action.value}. Tools: "
                        + (", ".join(tools_planned) if tools_planned else "none")
                    ),
                })
                for tool in deliberation.tools_to_use:
                    if tool.would_help and tool.rationale:
                        await progress("thinking", {
                            "step_type": "plan",
                            "content": f"[{tool.tool_name}] {tool.rationale}",
                        })

            # 3. Execute based on deliberation decision
            response: str
            tools_used: list[str] = []
            action_result: dict[str, Any] | None = None

            # Debate trigger (evaluated once; used in both USE_TOOLS and
            # RESPOND_DIRECTLY branches). Triggered when:
            #   (a) `force_debate` is set in task.params, OR
            #   (b) `auto_debate` is set AND the query is evaluative
            #       (compare / vs / tradeoffs / pros-and-cons / ...).
            auto_debate = task.params.get("auto_debate", False)
            force_debate = task.params.get("force_debate", False)
            should_debate = force_debate or (
                auto_debate and _is_evaluative_query(query)
            )
            debate_available = should_debate and "debate" in self.agents

            if deliberation.action == ActionDecision.RESPOND_DIRECTLY:
                response = await self._respond_from_context(
                    query, deliberation, user_model, context
                )
            elif deliberation.action == ActionDecision.USE_TOOLS:
                tool_results = await self._execute_selected_tools(
                    deliberation, allowed_sources=allowed_sources,
                    max_results=max_results, progress=progress,
                )
                tools_used = [t.tool_name for t in deliberation.tools_to_use if t.would_help]
                action_result = tool_results

                if progress:
                    counts = {
                        "papers": len(tool_results.get("papers") or []),
                        "repositories": len(tool_results.get("repositories") or []),
                        "models": len(tool_results.get("models") or []),
                        "web_results": len(tool_results.get("web_results") or []),
                    }
                    summary = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
                    await progress("thinking", {
                        "step_type": "action",
                        "content": f"Retrieved {summary or 'no results'}.",
                    })

                # 3a. Multi-round bull/bear debate on the EVIDENCE (not a
                # synthesized answer). Runs before synthesis so the debate's
                # strengths/concerns can inform the final response rather
                # than sitting as post-hoc side-car metadata.
                debate_out: dict[str, Any] | None = None
                if debate_available:
                    if progress:
                        await progress("stage", {
                            "stage": "debating",
                            "message": "Running bull/bear debate on research evidence...",
                        })
                    debate_out = await self._run_debate(
                        query, deliberation, action_result, synthesized_answer=None,
                    )
                    if debate_out:
                        action_result.update(debate_out)
                        if progress:
                            await progress("thinking", {
                                "step_type": "reflection",
                                "content": (
                                    f"Debate: {len(debate_out.get('bull_strengths', []))} strengths, "
                                    f"{len(debate_out.get('bear_concerns', []))} concerns, "
                                    f"confidence {debate_out.get('confidence_score', 0.0):.2f}"
                                ),
                            })

                # 3b. Synthesis — now informed by debate output when present.
                if progress:
                    await progress("stage", {
                        "stage": "synthesizing",
                        "message": "Synthesizing results...",
                    })
                    await progress("thinking", {
                        "step_type": "reflection",
                        "content": (
                            "Synthesizing a debate-informed answer that weighs bull strengths against bear concerns."
                            if debate_out
                            else "Synthesizing a structured answer with LaTeX formulas and IEEE-style algorithms where relevant."
                        ),
                    })
                    response = await self._synthesize_with_results_stream(
                        query, tool_results, user_model, context, deliberation,
                        progress, debate_output=debate_out,
                    )
                else:
                    response = await self._synthesize_with_results(
                        query, tool_results, user_model, context, deliberation,
                        debate_output=debate_out,
                    )

                # 3c. Optional refinement pass. When the debate's confidence
                # is low, re-synthesize once to explicitly address bear
                # concerns. Controllable via task.params["refine"] (default
                # True) and task.params["refine_threshold"] (default 0.7).
                if debate_out and task.params.get("refine", True):
                    threshold = float(task.params.get("refine_threshold", 0.7))
                    confidence = float(debate_out.get("confidence_score", 0.0))
                    if confidence < threshold:
                        if progress:
                            await progress("stage", {
                                "stage": "refining",
                                "message": (
                                    f"Refining (debate confidence {confidence:.2f} "
                                    f"< {threshold:.2f})..."
                                ),
                            })
                        # If refinement LLM call fails, keep the first draft
                        # rather than failing the whole query — a good draft
                        # is strictly better than an error response.
                        try:
                            response = await self._refine_synthesis(
                                query, response, debate_out,
                            )
                            action_result["refined"] = True
                            action_result["refine_reason"] = (
                                f"debate confidence {confidence:.2f} below threshold {threshold:.2f}"
                            )
                        except Exception as e:
                            self._logger.warning(
                                "refinement_failed_kept_draft",
                                error=str(e),
                                confidence=confidence,
                            )
                            action_result["refined"] = False
                            action_result["refine_error"] = str(e)
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
                        "intent_weights": deliberation.intent_weights,
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
        progress: ProgressCallback = None,
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

        # Filter to actionable tools
        builtin_sources = {"arxiv", "github", "huggingface", "hugging-face", "web"}
        active_tools: list[ToolConsideration] = []
        for tool in deliberation.tools_to_use:
            if not tool.would_help:
                continue
            if allowed_set is not None and tool.tool_name not in allowed_set:
                if tool.tool_name in builtin_sources:
                    self._logger.info(
                        "tool_skipped_by_source_filter",
                        tool=tool.tool_name,
                        allowed=list(allowed_set),
                    )
                    continue
            active_tools.append(tool)

        async def _run_tool(tool: ToolConsideration) -> tuple[str, Any]:
            """Execute a single tool and return (result_key, data)."""
            search_query = _enrich_query_with_date(tool.search_query)
            self._logger.info(
                "executing_tool",
                tool=tool.tool_name,
                query=search_query,
                rationale=tool.rationale,
            )
            if progress:
                await progress("log", {
                    "timestamp": time.time(),
                    "step": f"{tool.tool_name}_search",
                    "status": "started",
                    "elapsed_ms": 0,
                })
                await progress("stage", {
                    "stage": "searching",
                    "tool": tool.tool_name,
                    "message": f"Searching {tool.tool_name}...",
                })

            tool_start = time.time()
            try:
                if tool.tool_name == "arxiv":
                    result_key, data = "papers", await self._search_arxiv(search_query, max_results=limit)
                elif tool.tool_name == "github":
                    result_key, data = "repositories", await self._search_github(search_query, max_results=limit)
                elif tool.tool_name in ("huggingface", "hugging-face"):
                    result_key, data = "huggingface", await self._search_huggingface(search_query, max_results=limit)
                elif tool.tool_name == "web":
                    result_key, data = "web_results", await self._search_web(search_query, max_results=limit)
                elif tool.tool_name == "image":
                    enhanced_prompt = await self._enhance_image_prompt(tool.search_query)
                    result_key, data = "images", await self._generate_image(enhanced_prompt)
                else:
                    result_key, data = "mcp_results", await self._execute_mcp_tool(tool.tool_name, search_query)

                elapsed_ms = (time.time() - tool_start) * 1000
                if progress:
                    if isinstance(data, list):
                        items = data
                        count = len(data)
                    elif isinstance(data, dict):
                        items = data.get("models", []) + data.get("datasets", [])
                        count = len(items)
                    else:
                        items = []
                        count = 0
                    await progress("tool_complete", {
                        "tool": tool.tool_name,
                        "result_key": result_key,
                        "count": count,
                        "items": items,
                        "elapsed_ms": round(elapsed_ms),
                    })
                    await progress("log", {
                        "timestamp": time.time(),
                        "step": f"{tool.tool_name}_search",
                        "status": "completed",
                        "elapsed_ms": round(elapsed_ms),
                    })
                return (result_key, data)
            except Exception as exc:
                elapsed_ms = (time.time() - tool_start) * 1000
                if progress:
                    await progress("error", {
                        "message": str(exc),
                        "tool": tool.tool_name,
                    })
                    await progress("log", {
                        "timestamp": time.time(),
                        "step": f"{tool.tool_name}_search",
                        "status": "error",
                        "elapsed_ms": round(elapsed_ms),
                        "detail": str(exc),
                    })
                raise

        # Execute ALL tools in parallel
        tool_outputs = await asyncio.gather(
            *[_run_tool(t) for t in active_tools],
            return_exceptions=True,
        )

        # Aggregate results
        for i, output in enumerate(tool_outputs):
            if isinstance(output, Exception):
                self._logger.warning(
                    "tool_execution_error",
                    tool=active_tools[i].tool_name,
                    error=str(output),
                )
                continue
            key, data = output
            if key == "huggingface":
                results["models"].extend(data.get("models", []))
                results["datasets"].extend(data.get("datasets", []))
            elif key in results:
                results[key].extend(data if isinstance(data, list) else [data])

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

    async def _run_debate(
        self,
        query: str,
        deliberation: Deliberation,
        tool_results: dict[str, Any] | None,
        synthesized_answer: str | None = None,
    ) -> dict[str, Any] | None:
        """Invoke the registered DebateAgent on an evaluative query.

        Packages the research output (and optionally a synthesized answer)
        as context for the bull/bear debate, and flattens the result into
        the keys the API expects (bull_strengths, bear_concerns,
        recommendation, debate_history).

        When `synthesized_answer` is None, the debate runs on raw research
        evidence — this is the pre-synthesis path where the debate's output
        will feed into synthesis rather than commenting on it.
        """
        from dova.agents.base import AgentTask as _AgentTask

        debate_agent = self.agents.get("debate")
        if debate_agent is None:
            return None

        ctx: dict[str, Any] = {}
        if synthesized_answer:
            ctx["orchestrator_answer"] = synthesized_answer
        if tool_results:
            # Keep the context compact — the DebateAgent only needs a
            # summary of the evidence, not every raw field.
            ctx["papers"] = (tool_results.get("papers") or [])[:5]
            ctx["repositories"] = (tool_results.get("repositories") or [])[:5]
            ctx["models"] = (tool_results.get("models") or [])[:5]
            ctx["web_results"] = (tool_results.get("web_results") or [])[:5]

        try:
            result = await debate_agent.execute(
                _AgentTask(type="debate", params={"topic": query, "context": ctx})
            )
        except Exception as e:
            self._logger.warning("debate_execution_error", error=str(e))
            return None

        if not result.success or not result.data:
            self._logger.warning(
                "debate_failed",
                error=result.error if not result.success else "empty_data",
            )
            return None

        d = result.data
        return {
            "bull_strengths": d.get("bull_strengths", []),
            "bear_concerns": d.get("bear_concerns", []),
            "balanced_assessment": d.get("balanced_assessment", ""),
            "recommendation": d.get("recommendation", ""),
            "confidence_score": d.get("confidence_score", 0.0),
            "debate_summary": d.get("summary", ""),
        }

    async def _execute_mcp_tool(self, tool_name: str, query: str) -> list[dict[str, Any]]:
        """
        Execute an MCP tool by name.

        For aggregated tools like "awslabs", selects the best matching server.
        For the "bio" umbrella, may fan out to multiple sub-servers in parallel
        when the query has keyword signals across multiple biomed domains
        (e.g., "phase-III sofosbuvir trials" → both trials + chemical compound).
        """
        results: list[dict[str, Any]] = []

        # Get matching MCP servers for this tool
        servers = get_mcp_servers_for_tool(tool_name)

        if not servers:
            self._logger.warning("no_mcp_servers_for_tool", tool=tool_name)
            return results

        # Semantic multi-select for the bio umbrella: run every sub-server
        # whose keyword score is positive, so cross-domain biomed queries
        # get literature + trial + compound context simultaneously.
        if tool_name == "bio":
            selected = self._select_bio_servers(servers, query)
            self._logger.info(
                "executing_bio_fanout",
                tool=tool_name,
                servers=selected,
                query=query,
            )
            async def _run(name: str) -> dict[str, Any] | None:
                call = self._get_mcp_tool_for_query(name, query)
                params = self._get_mcp_tool_params(name, call, query)
                self._logger.info(
                    "bio_server_call_starting",
                    server=name,
                    tool=call,
                    params_keys=list(params.keys()),
                )
                try:
                    r = await self.call_tool(name, call, params)
                    if r.success and r.data:
                        # Report size so operators can see the call really fired.
                        if isinstance(r.data, str):
                            data_size = f"{len(r.data)} chars"
                        elif isinstance(r.data, list):
                            data_size = f"{len(r.data)} items"
                        elif isinstance(r.data, dict):
                            data_size = f"dict keys={list(r.data.keys())[:5]}"
                        else:
                            data_size = type(r.data).__name__
                        self._logger.info(
                            "bio_server_call_complete",
                            server=name,
                            tool=call,
                            success=True,
                            data=data_size,
                        )
                        return {"source": name, "tool": call, "data": r.data}
                    self._logger.warning(
                        "bio_server_call_empty",
                        server=name,
                        tool=call,
                        success=r.success,
                        error=r.error,
                    )
                except Exception as e:
                    self._logger.warning(
                        "bio_server_call_exception",
                        server=name,
                        tool=call,
                        error=str(e),
                    )
                return None

            fan_results = await asyncio.gather(*[_run(s) for s in selected])
            return [r for r in fan_results if r is not None]

        # Single-best selection for non-bio aggregates (e.g., awslabs).
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

    def _select_bio_servers(self, servers: list[str], query: str) -> list[str]:
        """
        Semantic multi-select for the bio umbrella.

        Returns every bio sub-server whose keyword signal is positive, so a
        multi-domain query (e.g., "phase-III sofosbuvir trials for hepatitis")
        hits PubMed + ClinicalTrials + PubChem in parallel. If no keywords
        match at all, returns the default literature server only.
        """
        query_lower = query.lower()
        bio_keywords: dict[str, list[str]] = {
            "clinicaltrials-bio": [
                # ClinicalTrials.gov-specific signals
                "clinical trial", "clinicaltrials", "nct", "trial",
                "recruiting", "enrollment", "phase i", "phase ii",
                "phase iii", "phase iv", "randomized", "placebo", "cohort",
                "double-blind", "eligibility", "primary endpoint",
                "secondary endpoint", "adaptive design", "ind-enabling",
                "ind enabling", "investigational new drug",
            ],
            "pubchem-bio": [
                # Small-molecule / cheminformatics signals
                "compound", "molecule", "small molecule", "drug structure",
                "smiles", "inchi", "chemical formula", "pubchem", "cid",
                "bioassay", "ghs hazard", "substructure", "superstructure",
                "cheminformatics", "admet", "logp", "herg", "cyp",
                "pharmacokinetic", "pharmacodynamic", "drug-drug interaction",
                "ddi", "hepatotoxicity", "retrosynthesis",
            ],
            "pubmed-bio": [
                # Literature / biomedical research signals — broad so most
                # biotech/pharma questions pick up PubMed as at least one
                # of the servers to call.
                "pubmed", "pmid", "pmc", "mesh", "biomedical literature",
                "medline", "clinical study", "systematic review",
                "meta-analysis", "case report", "cohort study",
                "article abstract", "citation",
                # General biomed vocabulary — proteins, genes, targets, diseases
                "protein", "peptide", "antibody", "antigen", "epitope",
                "binder", "binding affinity", "affinity", "kinetics",
                "structure-based", "cryptic site", "docking", "alphafold",
                "af2", "rfdiffusion", "de novo design", "protein design",
                "enzyme", "catalysis", "directed evolution",
                "gene", "genome", "genomic", "gwas", "variant",
                "mutation", "allele", "expression", "transcriptom",
                "proteomic", "metabolomic", "single-cell", "scrna",
                "crispr", "gene editing", "base editing", "prime editing",
                "aav", "lnp", "vector",
                "cancer", "oncolog", "tumor", "tumour", "melanoma",
                "leukemia", "lymphoma", "diabetes", "alzheimer",
                "parkinson", "hepatitis", "hiv", "sars-cov", "covid",
                "rare disease", "therapeutic", "biomarker",
                "drug discovery", "lead optimization", "hit rate",
                "off-target", "target discovery", "mechanism of action",
                "ehr", "real-world evidence", "rwe", "clinical note",
                "pathology", "histopathology", "radiology",
            ],
        }

        hits: list[str] = [
            s for s in servers
            if any(kw in query_lower for kw in bio_keywords.get(s, []))
        ]
        if hits:
            return hits

        # No explicit keywords — fall back to literature (broadest entry point).
        if "pubmed-bio" in servers:
            return ["pubmed-bio"]
        return servers[:1]

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
            # Bio umbrella — keyword routing to the most specific server.
            "clinicaltrials-bio": [
                "clinical trial", "clinicaltrials", "nct", "trial", "recruiting",
                "enrollment", "phase i", "phase ii", "phase iii", "phase iv",
                "randomized", "placebo", "cohort", "double-blind", "eligibility",
            ],
            "pubchem-bio": [
                "compound", "molecule", "drug structure", "smiles", "inchi",
                "chemical formula", "pubchem", "cid", "bioassay", "ghs hazard",
                "substructure", "superstructure", "cheminformatics",
            ],
            "pubmed-bio": [
                "pubmed", "pmid", "pmc", "mesh", "biomedical literature",
                "medline", "clinical study", "systematic review", "meta-analysis",
                "case report", "cohort study", "article abstract", "citation",
            ],
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

        # Default to PubMed for general biomedical queries if no specific match.
        # Literature search is the broadest starting point across the bio umbrella.
        if best_score == 0 and "pubmed-bio" in servers:
            return "pubmed-bio"

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
            # Bio servers — primary search tool for each
            "pubmed-bio": "pubmed_search_articles",
            "clinicaltrials-bio": "clinicaltrials_search_studies",
            "pubchem-bio": "pubchem_search_compounds",
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

        # Bio servers — each has its own schema (verified against live endpoints).
        if server_name == "pubmed-bio":
            # pubmed_search_articles: {query, maxResults, ...}
            return {"query": query, "maxResults": 10}
        if server_name == "clinicaltrials-bio":
            # clinicaltrials_search_studies: free-text `query`, no max_results in schema.
            return {"query": query}
        if server_name == "pubchem-bio":
            # pubchem_search_compounds: searchType + identifierType + identifiers[].
            # Default to name-based lookup — best fit for free-text biomed queries.
            return {
                "searchType": "identifier",
                "identifierType": "name",
                "identifiers": [query],
            }

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
        if not result.success:
            self._logger.warning("arxiv_search_failed", error=result.error, query=query)
            return []
        if not result.data:
            self._logger.warning("arxiv_search_empty", query=query)
            return []

        data = result.data
        if isinstance(data, dict) and "papers" in data:
            papers = data["papers"]
        elif isinstance(data, list):
            papers = data
        else:
            papers = [data]

        def _arxiv_url(p: dict) -> str:
            url = p.get("url", "")
            if url and url.startswith("http"):
                return url
            arxiv_id = p.get("id", "")
            if arxiv_id:
                return f"https://arxiv.org/abs/{arxiv_id}"
            return url or ""

        return [
            {
                "title": p.get("title", ""),
                "url": _arxiv_url(p),
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
        if not result.success:
            self._logger.warning("github_search_failed", error=result.error, query=query)
            return []
        if not result.data:
            self._logger.warning("github_search_empty", query=query)
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

    @staticmethod
    def _parse_hf_markdown(text: str) -> list[dict[str, Any]]:
        """Parse HuggingFace MCP markdown response into structured dicts.

        The HF MCP server returns results as formatted markdown, not JSON.
        Two formats are used:

        hub_repo_search format:
            ### owner/model-name
            **Downloads:** 123 | **Likes:** 45 | **Trending Score:** 0.5
            **Tags:** tag1, tag2
            **Link:** [https://hf.co/owner/model-name](...)

        paper_search format:
            ## Paper Title Here
            Published on 6 Oct, 2025
            **Authors:** Author1, Author2
            ### Abstract
            Abstract text...
        """
        items: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        in_abstract = False

        for line in text.split("\n"):
            line = line.strip()

            # New item: ## heading (paper title) or ### heading (repo id)
            # But skip "### Abstract" which is a sub-section, not a new item
            if line.startswith("## ") and not line.startswith("### "):
                # h2 = paper title or section header
                title = line[3:].strip()
                # Skip section headers like "## Models (5)"
                if re.match(r"^(Models|Datasets|Spaces)\s*\(\d+\)", title):
                    continue
                if current:
                    items.append(current)
                current = {"id": title, "title": title, "name": title}
                in_abstract = False
                continue

            if line.startswith("### "):
                heading = line[4:].strip()
                if heading.lower() == "abstract":
                    in_abstract = True
                    if current:
                        current.setdefault("description", "")
                    continue
                # h3 that's not "Abstract" = repo item (hub_repo_search format)
                in_abstract = False
                if current:
                    items.append(current)
                current = {"id": heading, "title": heading, "name": heading}
                continue

            if current is None:
                continue

            # Collect abstract text
            if in_abstract and line and not line.startswith("**"):
                existing = current.get("description", "")
                current["description"] = (existing + " " + line).strip()[:500]
                continue

            if line.startswith("---"):
                in_abstract = False
                continue

            # Parse **Key:** value patterns
            if "**Downloads:**" in line:
                for part in line.split("|"):
                    part = part.strip()
                    if "**Downloads:**" in part:
                        try:
                            current["downloads"] = int(part.split("**Downloads:**")[1].strip().replace(",", ""))
                        except (ValueError, IndexError):
                            pass
                    elif "**Likes:**" in part:
                        try:
                            current["likes"] = int(part.split("**Likes:**")[1].strip().replace(",", ""))
                        except (ValueError, IndexError):
                            pass

            elif "**Tags:**" in line:
                tags_str = line.split("**Tags:**")[1].strip()
                current["tags"] = [t.strip() for t in tags_str.split(",") if t.strip()]
                for tag in current["tags"]:
                    if tag in ("text-generation", "fill-mask", "text-classification",
                               "token-classification", "question-answering", "summarization",
                               "translation", "image-classification", "object-detection",
                               "text-to-image", "automatic-speech-recognition"):
                        current["pipeline_tag"] = tag
                        break

            elif "**Authors:**" in line:
                current["authors"] = line.split("**Authors:**")[1].strip()

            elif "**Link:**" in line:
                url_match = re.search(r'\[?(https?://[^\s\]]+)', line)
                if url_match:
                    current["url"] = url_match.group(1)

            elif line.startswith("Published on"):
                current["published"] = line.replace("Published on", "").strip()

            elif "**Created:**" in line:
                current["last_modified"] = line.split("**Created:**")[1].strip()

        if current:
            items.append(current)

        return items

    async def _search_huggingface(self, query: str, max_results: int = 10) -> dict[str, list[dict[str, Any]]]:
        """Search HuggingFace for models and datasets using hub_repo_search."""
        results: dict[str, list[dict[str, Any]]] = {"models": [], "datasets": []}

        # hub_repo_search returns both models and datasets; also search papers
        model_result, paper_result = await asyncio.gather(
            self.search_huggingface(query, search_type="models", limit=max_results),
            self.search_huggingface(query, search_type="papers", limit=max(max_results // 2, 5)),
        )

        if not model_result.success:
            self._logger.warning("huggingface_model_search_failed", error=model_result.error, query=query)
        if not paper_result.success:
            self._logger.warning("huggingface_paper_search_failed", error=paper_result.error, query=query)

        if model_result.success and model_result.data:
            data = model_result.data
            # HF MCP server returns markdown text, not JSON
            if isinstance(data, str):
                items = self._parse_hf_markdown(data)
            elif isinstance(data, list):
                items = [m for m in data if isinstance(m, dict)]
            elif isinstance(data, dict):
                items = [data]
            else:
                items = []

            for m in items:
                model_id = m.get("id", m.get("modelId", ""))
                entry = {
                    "id": model_id,
                    "title": m.get("title", model_id),
                    "name": model_id,
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                    "tags": m.get("tags", []),
                    "pipeline_tag": m.get("pipeline_tag", ""),
                    "library_name": m.get("library_name", ""),
                    "url": m.get("url", f"https://huggingface.co/{model_id}" if model_id else ""),
                    "last_modified": m.get("lastModified", m.get("last_modified", "")),
                }
                results["models"].append(entry)

        # Paper results (also markdown text)
        if paper_result.success and paper_result.data:
            data = paper_result.data
            if isinstance(data, str):
                papers = self._parse_hf_markdown(data)
            elif isinstance(data, list):
                papers = [p for p in data if isinstance(p, dict)]
            elif isinstance(data, dict):
                papers = [data]
            else:
                papers = []

            for p in papers:
                paper_id = p.get("id", p.get("paperId", ""))
                results["datasets"].append({
                    "id": paper_id,
                    "title": p.get("title", paper_id),
                    "name": paper_id,
                    "downloads": 0,
                    "url": p.get("url", ""),
                    "source": "hf_papers",
                })

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
        debate_output: dict[str, Any] | None = None,
    ) -> str:
        """Synthesize results into a personalized response."""
        prompt = self._build_synthesis_prompt(
            query, results, user_model, context, deliberation,
            debate_output=debate_output,
        )
        return await self.think(
            prompt,
            task_type=TaskType.SUMMARIZATION,
            temperature=0.5,
        )

    async def _synthesize_with_results_stream(
        self,
        query: str,
        results: dict[str, Any],
        user_model: UserModel,
        context: ConversationContext,
        deliberation: Deliberation,
        progress: Any,
        debate_output: dict[str, Any] | None = None,
    ) -> str:
        """Synthesize results with token streaming via progress callback.

        Builds the same prompt as _synthesize_with_results, but streams tokens
        as synthesis_token events so the frontend can display them incrementally.
        Returns the full response string.
        """
        prompt = self._build_synthesis_prompt(
            query, results, user_model, context, deliberation,
            debate_output=debate_output,
        )

        chunks: list[str] = []
        async for token in self.think_stream(
            prompt,
            task_type=TaskType.SUMMARIZATION,
            temperature=0.5,
        ):
            chunks.append(token)
            await progress("synthesis_token", {"token": token})

        return "".join(chunks)

    async def _refine_synthesis(
        self,
        query: str,
        initial_response: str,
        debate_output: dict[str, Any],
    ) -> str:
        """One-pass refinement that addresses bear concerns explicitly.

        Triggered when the debate's confidence_score is below the refinement
        threshold. The refinement prompt shows the initial draft alongside
        the bear concerns and asks for a revised answer that either
        incorporates the concerns as caveats or rebuts them with evidence.
        """
        bull = debate_output.get("bull_strengths", []) or []
        bear = debate_output.get("bear_concerns", []) or []
        balanced = debate_output.get("balanced_assessment", "") or ""
        recommendation = debate_output.get("recommendation", "") or ""

        bull_text = "\n".join(f"  - {s}" for s in bull) or "  (none recorded)"
        bear_text = "\n".join(f"  - {c}" for c in bear) or "  (none recorded)"

        refine_prompt = f"""Revise your previous answer to directly address the adversarial critique below.

Original user question: {query}

YOUR PREVIOUS DRAFT:
{initial_response}

ADVERSARIAL CRITIQUE (Bull vs Bear debate):

Bull strengths (points the draft should preserve):
{bull_text}

Bear concerns (weaknesses the draft must address — either rebut with evidence or acknowledge as caveats):
{bear_text}

Balanced assessment from moderator:
{balanced}

Moderator recommendation:
{recommendation}

REFINEMENT RULES:
1. Keep the draft's correct content and structure.
2. For EACH bear concern, either:
   (a) rebut it using evidence from the original research results, OR
   (b) acknowledge it explicitly as a caveat / limitation in the answer.
3. Do not fabricate new sources. Use only what the draft already cited.
4. Preserve formatting: headings, LaTeX math, algorithm blocks, inline source URLs.
5. The revised answer should read as a single coherent response — do NOT include
   meta-commentary like "In response to the critique" or "Addressing concerns".
6. Keep length comparable to the original draft (do not bloat).

Return ONLY the revised answer."""

        return await self.think(
            refine_prompt,
            task_type=TaskType.SUMMARIZATION,
            temperature=0.4,
        )

    def _build_synthesis_prompt(
        self,
        query: str,
        results: dict[str, Any],
        user_model: UserModel,
        context: ConversationContext,
        deliberation: Deliberation,
        debate_output: dict[str, Any] | None = None,
    ) -> str:
        """Build the synthesis prompt. Shared by streaming and non-streaming paths.

        Result aggregation is weighted by the deliberation's semantic intent
        distribution across {ai, bio, web}. A fixed budget of ~15 slots is
        divided proportionally so a 60% AI / 30% Bio / 10% Web query surfaces
        roughly 9 AI items (split across papers/repos/models), 5 bio items,
        and 2 web items to the synthesis LLM. Every allocated group gets a
        minimum of 2 items when it has any results, so no group is starved.

        When `debate_output` is provided, an "Adversarial Analysis" section
        is injected into the prompt so the synthesis LLM weighs bull
        strengths against bear concerns when producing the final answer.
        """
        result_parts = []

        weights = deliberation.intent_weights or {"ai": 0.5, "bio": 0.2, "web": 0.3}
        total_budget = 15

        def _slots(weight: float) -> int:
            return max(2, round(weight * total_budget))

        ai_slots = _slots(weights.get("ai", 0.0))
        bio_slots = _slots(weights.get("bio", 0.0))
        web_slots = _slots(weights.get("web", 0.0))

        # AI group is split across papers/repos/models. Divide as evenly
        # as possible but give papers the remainder (they're usually richer).
        num_ai_channels = sum(
            1 for k in ("papers", "repositories", "models") if results.get(k)
        ) or 1
        per_channel = max(2, ai_slots // num_ai_channels)

        papers_cap = per_channel if results.get("papers") else 0
        repos_cap = per_channel if results.get("repositories") else 0
        models_cap = per_channel if results.get("models") else 0
        # Award any remainder to papers.
        if results.get("papers"):
            papers_cap += ai_slots - per_channel * num_ai_channels

        if results.get("papers") and papers_cap > 0:
            papers = results["papers"][:papers_cap]
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
            result_parts.append(
                f"**Papers Found ({len(results['papers'])}, showing {len(papers)} "
                f"per {int(weights.get('ai', 0) * 100)}% AI weight):**\n"
                + "\n".join(paper_lines)
            )

        if results.get("repositories") and repos_cap > 0:
            repos = results["repositories"][:repos_cap]
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
            result_parts.append(
                f"**Repositories Found ({len(results['repositories'])}, showing {len(repos)}):**\n"
                + "\n".join(repo_lines)
            )

        if results.get("models") and models_cap > 0:
            models = results["models"][:models_cap]
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
            result_parts.append(
                f"**Models Found ({len(results['models'])}, showing {len(models)}):**\n"
                + "\n".join(model_lines)
            )

        if results.get("web_results") and web_slots > 0:
            web = results["web_results"][:web_slots]
            web_lines = []
            for w in web:
                title = w.get('title', 'Unknown')
                desc = w.get('description', '')[:200]
                url = w.get('url', '')
                line = f"- **{title}**: {desc}"
                if url:
                    line += f"\n  URL: {url}"
                web_lines.append(line)
            result_parts.append(
                f"**Web Results ({len(results['web_results'])}, showing {len(web)} "
                f"per {int(weights.get('web', 0) * 100)}% Web weight):**\n"
                + "\n".join(web_lines)
            )

        if results.get("images"):
            images = results["images"]
            result_parts.append(f"**Images Generated ({len(images)}):** Images have been created based on your request.")

        if results.get("mcp_results"):
            # Truncate each bio source's payload proportionally to the bio
            # weight. A 10% bio query gets ~400 chars per server; a 60% bio
            # query gets ~2000 chars so the synthesis LLM sees full context.
            bio_budget_chars = max(400, int(2500 * weights.get("bio", 0.2)))
            for mcp_result in results["mcp_results"]:
                source = mcp_result.get("source", "unknown")
                data = mcp_result.get("data", {})
                if "pricing" in source.lower() and isinstance(data, (dict, list)):
                    fuzzy_result = self._process_pricing_with_fuzzy_match(query, data)
                    if fuzzy_result:
                        result_parts.append(fuzzy_result)
                        continue
                is_bio = source in ("pubmed-bio", "clinicaltrials-bio", "pubchem-bio")
                cap = bio_budget_chars if is_bio else 500
                if isinstance(data, dict):
                    summary = json.dumps(data, indent=2)[:cap]
                elif isinstance(data, list):
                    summary = f"{len(data)} items returned"
                else:
                    summary = str(data)[:cap]
                header = f"**{source}"
                if is_bio:
                    header += f" ({int(weights.get('bio', 0) * 100)}% Bio weight)"
                header += ":**"
                result_parts.append(f"{header}\n{summary}")

        style_instructions = self._get_style_instructions(user_model)

        inferred_context = ""
        if hasattr(deliberation, 'inferred_entity') and deliberation.inferred_entity:
            inferred_context = f"\nNote: User likely meant '{deliberation.inferred_entity}' - adjust response accordingly.\n"

        # Human-readable intent distribution for the synthesis LLM.
        weights_str = ", ".join(
            f"{int(v * 100)}% {k.upper()}" for k, v in weights.items() if v > 0
        ) or "balanced"

        # Adversarial analysis block — injected only when a bull/bear debate
        # has already run on the evidence. Formatted as compact bullets so it
        # adds minimal tokens while still steering the synthesis LLM toward a
        # balanced answer that both leverages strengths and addresses concerns.
        debate_block = ""
        if debate_output:
            bull = debate_output.get("bull_strengths", []) or []
            bear = debate_output.get("bear_concerns", []) or []
            balanced = debate_output.get("balanced_assessment", "") or ""
            recommendation = debate_output.get("recommendation", "") or ""
            confidence = debate_output.get("confidence_score", 0.0)

            bull_text = "\n".join(f"  - {s}" for s in bull) if bull else "  (none)"
            bear_text = "\n".join(f"  - {c}" for c in bear) if bear else "  (none)"

            debate_block = f"""
Adversarial Analysis (Bull vs Bear multi-round debate on the evidence above):

Bull strengths (use these to support claims in your answer):
{bull_text}

Bear concerns (address EVERY concern — rebut with evidence if possible, otherwise acknowledge as a caveat):
{bear_text}

Moderator balanced assessment:
{balanced or "  (none provided)"}

Moderator recommendation:
{recommendation or "  (none provided)"}

Debate confidence: {confidence:.2f}

DEBATE INTEGRATION RULES:
- Your answer MUST weigh strengths against concerns, not ignore either side.
- For each bear concern relevant to the question, explicitly acknowledge it
  (e.g., as a "limitation", "caveat", "open question", or "risk") OR rebut it
  with specific evidence from the search results.
- Do NOT include meta-commentary like "the bull says X and the bear says Y" —
  the debate informs the answer but is not the subject of the answer.
- If the moderator's recommendation conflicts with the raw evidence, trust the
  evidence and note the conflict.
"""

        return f"""Synthesize these search results into a helpful response.

TODAY'S DATE: {_current_date_str()}

User Question: {query}
Understanding: {deliberation.understanding}
Semantic intent distribution: {weights_str}
(Weight the depth and prominence of each section in your answer accordingly —
a higher-weighted group should dominate the narrative; lower-weighted groups
provide corroborating context.)
{inferred_context}
Search Results:
{chr(10).join(result_parts) if result_parts else "No results found."}
{debate_block}
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
   - ALWAYS include source URLs as inline links for every paper, repo, model, or web source you reference
   - Structure the response with clear sections and headers when covering multiple aspects
   - For technical topics, explain key concepts, methods, and their significance
   - For each key finding or recommendation, explain the RATIONALE: why it matters, what problem it solves, or what makes it significant
   - When comparing approaches, explain trade-offs and under what conditions each excels
   - Connect findings to practical implications — who benefits and how

7. MATH FORMATTING (LaTeX / KaTeX — MANDATORY when any equation, metric, complexity bound, or quantitative relation appears)
   - Inline math MUST be wrapped in single dollars: $O(n \\log n)$, $\\mathcal{{L}}(\\theta) = -\\mathbb{{E}}[\\log p_\\theta(x)]$
   - Display math (one-liners or important equations) MUST be wrapped in double dollars on their own paragraph:
       $$ \\mathrm{{Attention}}(Q,K,V) = \\mathrm{{softmax}}\\!\\left( \\tfrac{{QK^\\top}}{{\\sqrt{{d_k}}}} \\right) V $$
   - Use proper LaTeX macros: \\mathbb, \\mathcal, \\mathrm, \\nabla, \\sum, \\int, \\frac, \\sqrt, \\hat, \\tilde, \\bar, \\in, \\subseteq, \\to, \\leftarrow, \\Rightarrow, \\forall, \\exists, \\approx, \\propto, \\sim
   - Always render loss functions, scaling laws, gradient updates, complexity bounds, probabilities, and any symbol involving subscripts/superscripts as LaTeX — never as plain text like "O(n log n)" or "theta_hat"

8. ALGORITHMS — MANDATORY IEEE-STYLE PSEUDOCODE
   - Whenever the response describes a procedure, training loop, optimization step, decoding strategy, search method, or any multi-step algorithm, render it as a fenced block with language tag `algorithm` using IEEE / \\usepackage{{algorithmic}} keywords.
   - Format each line as: KEYWORD <expr or description>. Keywords MUST be UPPERCASE and drawn from:
       \\Require   (preconditions / inputs)
       \\Ensure    (postconditions / outputs)
       \\State     (straight-line step)
       \\If  ... \\Then   ... \\ElsIf ... \\Then   ... \\Else   ... \\EndIf
       \\For <i = 1 to N> \\Do   ... \\EndFor
       \\While <cond> \\Do       ... \\EndWhile
       \\Repeat    ... \\Until <cond>
       \\Function <Name>(<args>) ... \\EndFunction
       \\Return   <expr>
       \\Comment{{<text>}}
   - Inline math inside any algorithm line MUST use single-dollar LaTeX: `\\State $\\theta \\gets \\theta - \\eta \\nabla_\\theta \\mathcal{{L}}$`.
   - Each algorithm block should open with a caption line: `\\Caption{{Algorithm N: <Short Title>}}` (N = 1, 2, ...).
   - Example:
     ```algorithm
     \\Caption{{Algorithm 1: Mini-batch SGD with Momentum}}
     \\Require dataset $\\mathcal{{D}}$, learning rate $\\eta$, momentum $\\beta$, batch size $B$
     \\Ensure trained parameters $\\theta$
     \\State Initialize $\\theta \\leftarrow \\theta_0$,\\; $v \\leftarrow 0$
     \\For{{$t = 1$ to $T$}}
         \\State Sample batch $\\mathcal{{B}}_t \\subset \\mathcal{{D}}$ with $|\\mathcal{{B}}_t| = B$
         \\State $g_t \\gets \\tfrac{{1}}{{B}} \\sum_{{x \\in \\mathcal{{B}}_t}} \\nabla_\\theta \\mathcal{{L}}(\\theta; x)$
         \\State $v \\gets \\beta\\, v + (1 - \\beta)\\, g_t$
         \\State $\\theta \\gets \\theta - \\eta\\, v$
     \\EndFor
     \\Return $\\theta$
     ```
   - Do NOT use Python, Java, or shell snippets for algorithms — only the `algorithm` fenced block above. Keep runnable code blocks (` ```python `, etc.) strictly for glue/setup examples, never for core methodology.

Response:"""

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
            instructions.append("When showing any algorithm or procedure, use IEEE-style pseudocode in a fenced ```algorithm block (with \\State, \\For, \\If, $LaTeX$ math) instead of Python or shell code.")

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
