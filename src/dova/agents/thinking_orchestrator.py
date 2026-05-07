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
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Awaitable, Callable

import structlog

from dova.agents.base import AgentResult, AgentTask, BaseAgent
from dova.agents.conversation_context import ConversationContext, ConversationTurn
from dova.agents.user_model import ExpertiseLevel, ResponseDepth, UserModel
from dova.config.mcp_servers import (
    BIO_MCP_SERVERS,
    MASTER_PAPER_MCP_NAME,
    MASTER_PAPER_MCP_SUBJECT_KEYWORDS,
    MASTER_PAPER_MCP_UMBRELLA_DEFAULT_SUBJECT,
    MASTER_PAPER_MCP_UMBRELLA_SUBJECTS,
    list_mcp_servers,
)
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


# Recency windows for AI vs Bio paper search. AI moves fast — 12-month window.
# Bio literature cycles slower (trials, reviews, meta-analyses) — 24 months.
AI_RECENCY_MONTHS = 12
BIO_RECENCY_MONTHS = 24


def _months_ago_date(months: int, fmt: str = "%Y-%m-%d") -> str:
    """Return today minus *months* months formatted per *fmt*.

    Uses a 30.44-day-per-month approximation — good enough for PubMed and
    arXiv which both accept any date on the requested calendar day.
    """
    today = datetime.now()
    delta_days = int(round(months * 30.44))
    target = today - timedelta(days=delta_days)
    return target.strftime(fmt)


# Token-budget accounting constants for the pipeline worst-case bound
# (matches the paper's Thm.1 calibration). Overridable via env:
#   DOVA_TPI       per-call provider output cap
#   DOVA_TK        thinking-tier cap for synthesis
#   DOVA_KAPPA     per-slot synthesis tokens
#   DOVA_BRIDGE_EPS  bridge gating threshold (epsilon_pi)
_DEFAULT_T_PI = int(os.environ.get("DOVA_TPI", "4096"))
_DEFAULT_T_K = int(os.environ.get("DOVA_TK", "65536"))
_DEFAULT_KAPPA = int(os.environ.get("DOVA_KAPPA", "2100"))
_DEFAULT_EPS_PI = float(os.environ.get("DOVA_BRIDGE_EPS", "0.10"))
_DEFAULT_SLOT_BUDGET = 15


def _estimate_token_bound(
    intent_weights: dict[str, float] | None,
    allowed_groups: set[str] | None,
    T_pi: int = _DEFAULT_T_PI,
    T_k: int = _DEFAULT_T_K,
    kappa: int = _DEFAULT_KAPPA,
    B: int = _DEFAULT_SLOT_BUDGET,
    eps_pi: float = _DEFAULT_EPS_PI,
) -> dict[str, Any]:
    """Worst-case per-query output-token estimate (Thm.1 of the paper).

    T(q) <= T_pi * (1 + |G'| + 1[bridge on]) + min(B*kappa, T_k)

    Computable before any LLM call. Returns a dict with the final estimate
    and the per-component breakdown so operators can reason about the
    dominant term.
    """
    weights = intent_weights or {}
    groups = allowed_groups if allowed_groups is not None else set()
    n_groups = max(1, len(groups))
    bridge_on = (
        weights.get("ai", 0.0) >= eps_pi
        and weights.get("bio", 0.0) >= eps_pi
    )
    pre_synthesis = T_pi * (1 + n_groups + (1 if bridge_on else 0))
    synthesis = min(B * kappa, T_k)
    total = pre_synthesis + synthesis
    return {
        "estimate": int(total),
        "pre_synthesis": int(pre_synthesis),
        "synthesis": int(synthesis),
        "bridge_on": bridge_on,
        "n_groups": n_groups,
        "inputs": {"T_pi": T_pi, "T_k": T_k, "kappa": kappa, "B": B, "eps_pi": eps_pi},
    }


# Stopwords and filler phrases that cause PubMed's NLM parser to yield 0 hits
# when embedded in long natural-language queries. These aren't medical terms,
# so dropping them preserves intent while improving recall.
_PUBMED_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "of", "for", "in", "on", "to", "with",
    "is", "are", "was", "were", "be", "been", "being", "at", "by", "from",
    "as", "about", "into", "over", "under",
    "latest", "newest", "recent", "new", "current", "modern", "emerging",
    "state-of-the-art", "cutting-edge", "novel",
    "trending", "now", "today", "this year", "last year", "next year",
    "review", "overview", "summary", "comparison", "study",
})


"""Curated bio→AI mechanism analogues.

When a query uses biological vocabulary (left side) we hint the synthesis
LLM to consider the paired AI construct (right side). Cheap prompt-level
lever — no new LLM call, no new tool.

Keep entries mechanistic, not metaphorical — each pairing should have a
documented engineering precedent.
"""
_BIO_TO_AI_REFRAMES: dict[str, str] = {
    "olfactory": "sparse distributed representations, mixture-of-experts routing, locality-sensitive hashing (fly olfactory circuit → Dasgupta et al. 2017)",
    "immune": "clonal selection, negative selection, affinity maturation (artificial immune systems), adversarial critics with diverse exemplars",
    "antibody": "high-dimensional nearest-neighbor search, contrastive learning with negative mining",
    "t-cell": "gated routing with co-stimulation signals, multi-signal authorization in agent tool-use",
    "neural adaptation": "layer normalization, short-term plasticity in RNNs, adaptive learning rates",
    "synaptic plasticity": "STDP-inspired local learning, Hebbian updates, meta-learning",
    "glial": "modulatory gating, learning-rate scheduling, value-based credit assignment",
    "predictive coding": "hierarchical top-down generative models, error-driven training, free-energy formulations",
    "cortical column": "shared-weight cortical micro-circuits, capsule networks",
    "dopamine": "temporal-difference reward signals, RLHF reward modeling",
    "memory consolidation": "experience replay, complementary learning systems (hippocampal → cortical)",
    "hippocampus": "episodic memory buffer, retrieval-augmented models, episodic control",
    "place cell": "grid-cell analogues, spatial embeddings, position encoding",
    "evolution": "evolutionary strategies, neuroevolution, novelty search, quality-diversity algorithms",
    "mutation": "random search, perturbation-based exploration",
    "homeostasis": "intrinsic normalization, target-entropy control, homeostatic plasticity",
    "attention": "transformer self-attention (explicit analog of selective attention literature)",
    "pathway": "computational graph with gated subnetworks",
    "biomarker": "feature-importance ranking, causal discovery, signature learning",
    "metabolic": "budget-aware inference, early-exit, cascaded models",
    "gene regulatory network": "gated graph neural networks, Boolean network dynamics",
    "protein folding": "geometric deep learning, equivariant networks, diffusion-based structure generation",
    "enzyme": "catalytic functions as differentiable modules, retrosynthesis planning",
    "epigenetic": "context-conditional modulation, fast weights, adapters",
}


def _select_bio_to_ai_reframes(query: str, bio_keywords_hit: list[str] | None = None) -> list[str]:
    """Return curated bio→AI analogue lines matching the query's vocabulary.

    At most 3 reframes are returned — more crowds the prompt.
    """
    q = query.lower()
    hits: list[str] = []
    seen: set[str] = set()
    for bio_term, ai_analog in _BIO_TO_AI_REFRAMES.items():
        if bio_term in q and bio_term not in seen:
            seen.add(bio_term)
            hits.append(f"- {bio_term} → {ai_analog}")
        if len(hits) >= 3:
            break
    return hits


def _distill_pubmed_query(query: str, max_terms: int = 8) -> str:
    """Reduce a natural-language query to a PubMed-friendly keyword phrase.

    PubMed's parser ANDs every token together, so long conjunctive queries
    return zero hits. Strip years, commas, stopwords and quantifier phrases;
    keep at most *max_terms* meaningful tokens.
    """
    q = query.lower()
    # Remove 4-digit years and year ranges.
    q = re.sub(r"\b(19|20)\d{2}\b(?:-\d{2,4})?", " ", q)
    # Remove punctuation that confuses the parser.
    q = re.sub(r"[,:;\?\!\"\(\)\[\]]", " ", q)
    # Tokenise and filter.
    kept: list[str] = []
    for tok in q.split():
        if len(tok) <= 1:
            continue
        if tok in _PUBMED_STOPWORDS:
            continue
        if tok.isdigit():
            continue
        kept.append(tok)
        if len(kept) >= max_terms:
            break
    return " ".join(kept) if kept else query.strip()


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
    # Semantic intent weights across {ai, bio, web}. Sum to 1.0.
    # Used downstream (a) to weight result aggregation in synthesis, and
    # (b) to gate master_paper_mcp umbrella fan-out in _collect_results
    # when a weight meets the activation threshold (see line ~1494).
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

            # Per-stage token accounting + worst-case budget estimate.
            # The estimate is computable before any downstream LLM call
            # (it consumes only intent weights and enabled groups).
            stage_tokens: dict[str, int] = {}
            # Capture deliberation cost (already ran): read the last
            # output-token count from the base-agent attribute set by
            # think(). Zero if the fast path skipped the LLM.
            stage_tokens["deliberation"] = int(getattr(self, "last_output_tokens", 0) or 0)
            token_budget = _estimate_token_bound(
                intent_weights=deliberation.intent_weights,
                allowed_groups=allowed_groups,
            )
            self._logger.info(
                "token_budget_estimate",
                estimate=token_budget["estimate"],
                pre_synthesis=token_budget["pre_synthesis"],
                synthesis=token_budget["synthesis"],
                bridge_on=token_budget["bridge_on"],
                n_groups=token_budget["n_groups"],
            )
            max_budget_env = os.environ.get("DOVA_MAX_TOKEN_BUDGET")
            if max_budget_env:
                try:
                    limit = int(max_budget_env)
                    if token_budget["estimate"] > limit:
                        self._logger.warning(
                            "token_budget_exceeds_limit",
                            estimate=token_budget["estimate"],
                            limit=limit,
                        )
                except ValueError:
                    pass

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
                stage_tokens["respond"] = int(getattr(self, "last_output_tokens", 0) or 0)
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

                # 3a'. Cross-domain bridge analysis (Axis 1 #1). Only fires when
                # the query spans AI and Bio meaningfully AND both groups
                # returned evidence. Produces structured candidate bridges
                # (ai_method ↔ bio_target) that are rendered in the synthesis
                # prompt and critiqued by the debate step that follows.
                bridges = await self._analyze_cross_domain(
                    query, deliberation, tool_results, progress=progress,
                )
                stage_tokens["bridge"] = int(getattr(self, "last_output_tokens", 0) or 0) if bridges else 0
                if bridges:
                    action_result["cross_domain_bridges"] = bridges

                # 3a''. Drug-story chaining (Axis 1 #3). No LLM call — pure
                # string processing over pubchem/pubmed/trials MCP payloads.
                drug_story = self._extract_drug_story(
                    tool_results.get("mcp_results") or [], query,
                )
                if drug_story:
                    action_result["drug_story"] = drug_story
                    self._logger.info(
                        "drug_story_chained",
                        compound=drug_story.get("compound"),
                        pmids=len(drug_story.get("mechanism_pmids", [])),
                        ncts=len(drug_story.get("trial_nct_ids", [])),
                    )

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
                    # Streaming path bypasses last_output_tokens; estimate
                    # from response length at ~4 chars/token.
                    stage_tokens["synthesis"] = max(1, len(response) // 4)
                else:
                    response = await self._synthesize_with_results(
                        query, tool_results, user_model, context, deliberation,
                        debate_output=debate_out,
                    )
                    stage_tokens["synthesis"] = int(getattr(self, "last_output_tokens", 0) or 0)

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
                            stage_tokens["refine"] = int(getattr(self, "last_output_tokens", 0) or 0)
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

            # Attach budget + stage-token observability to action_result so
            # downstream response builders surface them through
            # extract_research_data() / chat.py metadata.
            if action_result is None:
                action_result = {}
            action_result["token_budget_estimate"] = token_budget
            action_result["stage_tokens"] = stage_tokens
            action_result["stage_tokens_total"] = sum(stage_tokens.values())
            self._logger.info(
                "pipeline_token_summary",
                total=action_result["stage_tokens_total"],
                stages=stage_tokens,
                budget_estimate=token_budget["estimate"],
            )

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
                async def _dispatch() -> tuple[str, Any]:
                    if tool.tool_name == "arxiv":
                        return "papers", await self._search_arxiv(search_query, max_results=limit)
                    if tool.tool_name == "github":
                        return "repositories", await self._search_github(search_query, max_results=limit)
                    if tool.tool_name in ("huggingface", "hugging-face"):
                        return "huggingface", await self._search_huggingface(search_query, max_results=limit)
                    if tool.tool_name == "web":
                        return "web_results", await self._search_web(search_query, max_results=limit)
                    if tool.tool_name == "image":
                        enhanced_prompt = await self._enhance_image_prompt(tool.search_query)
                        return "images", await self._generate_image(enhanced_prompt)
                    return "mcp_results", await self._execute_mcp_tool(tool.tool_name, search_query)

                try:
                    result_key, data = await asyncio.wait_for(
                        _dispatch(), timeout=self._TOOL_EXEC_TIMEOUT_S
                    )
                except asyncio.TimeoutError:
                    self._logger.warning(
                        "tool_execution_timeout",
                        tool=tool.tool_name,
                        timeout_s=self._TOOL_EXEC_TIMEOUT_S,
                    )
                    # Return empty so the gather result slot is well-formed
                    # and downstream aggregation doesn't see an exception.
                    fallback_key = {
                        "arxiv": "papers",
                        "github": "repositories",
                        "huggingface": "huggingface",
                        "hugging-face": "huggingface",
                        "web": "web_results",
                        "image": "images",
                    }.get(tool.tool_name, "mcp_results")
                    fallback_data: Any = (
                        {"models": [], "datasets": []}
                        if fallback_key == "huggingface"
                        else []
                    )
                    result_key, data = fallback_key, fallback_data

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

        # Determine which umbrellas (ai/bio/web) should *additionally* fan
        # out to master_paper_mcp. An umbrella is active when a tool in that
        # umbrella was selected by deliberation OR the deliberation's
        # intent_weights exceed the activation threshold (0.25 — above the
        # 10% web floor in v1.7 so pure-ai / pure-bio queries don't drag web
        # shards into the fan-out).
        #
        # The map covers every tool name that can appear in active_tools:
        # get_available_tools (line ~63) only surfaces umbrella-level names
        # to the deliberation LLM, so "bio" here transparently covers
        # pubmed-bio / clinicaltrials-bio / pubchem-bio. `image` and
        # `awslabs` are not paper sources and are correctly absent.
        _umbrella_by_tool = {
            "arxiv": "ai", "github": "ai", "huggingface": "ai", "hugging-face": "ai",
            "bio": "bio",
            "web": "web",
        }
        umbrellas_present: set[str] = set()
        for t in active_tools:
            u = _umbrella_by_tool.get(t.tool_name)
            if u and u in MASTER_PAPER_MCP_UMBRELLA_SUBJECTS:
                umbrellas_present.add(u)
        for u, w in (deliberation.intent_weights or {}).items():
            if u in MASTER_PAPER_MCP_UMBRELLA_SUBJECTS and w >= 0.25:
                umbrellas_present.add(u)

        def _master_query_for(umbrella: str) -> str:
            """Pick the representative search query for this umbrella."""
            for t in active_tools:
                if _umbrella_by_tool.get(t.tool_name) == umbrella and t.search_query:
                    return t.search_query
            if active_tools and active_tools[0].search_query:
                return active_tools[0].search_query
            return ""

        # Rank subjects across all active umbrellas by query-keyword overlap,
        # then cap the total number of master_paper_mcp shards per request so
        # a wide-umbrella query can't spawn 9 parallel calls.
        umbrella_subjects: dict[str, list[str]] = {}
        for u in sorted(umbrellas_present):
            umbrella_subjects[u] = self._select_master_subjects(
                u,
                _master_query_for(u),
                max_per_umbrella=self._MASTER_MAX_SUBJECTS_PER_UMBRELLA,
            )
        total = sum(len(s) for s in umbrella_subjects.values())
        if total > self._MASTER_MAX_SUBJECTS_PER_REQUEST:
            # Trim proportionally: keep the first N (already ranked) per
            # umbrella until we fit under the global cap.
            budget = self._MASTER_MAX_SUBJECTS_PER_REQUEST
            trimmed: dict[str, list[str]] = {}
            for u, subs in umbrella_subjects.items():
                take = max(1, budget // max(1, len(umbrella_subjects)))
                trimmed[u] = subs[:take]
                budget -= len(trimmed[u])
                if budget <= 0:
                    break
            umbrella_subjects = {u: s for u, s in trimmed.items() if s}

        # Split `limit` across active umbrellas proportional to intent
        # weights, so a low-weight umbrella doesn't burn MCP calls whose
        # results synthesis will trim in _allocate_top_papers. Floor of 1
        # per active umbrella preserves the activation signal.
        _intent_weights = deliberation.intent_weights or {}
        _active_weight_total = sum(
            max(0.0, float(_intent_weights.get(u, 0.0))) for u in umbrella_subjects
        )

        def _limit_for(umbrella: str) -> int:
            if _active_weight_total > 0:
                w = max(0.0, float(_intent_weights.get(umbrella, 0.0)))
                share = w / _active_weight_total
            else:
                share = 1.0 / max(len(umbrella_subjects), 1)
            return max(1, round(share * limit))

        async def _run_master(umbrella: str) -> tuple[str, list[dict[str, Any]]]:
            data = await self._invoke_master_paper_mcp(
                umbrella=umbrella,
                subjects=umbrella_subjects.get(umbrella, []),
                query=_enrich_query_with_date(_master_query_for(umbrella)),
                limit=_limit_for(umbrella),
                progress=progress,
            )
            return ("mcp_results", data)

        master_tasks = [_run_master(u) for u in umbrella_subjects.keys()]
        master_umbrellas_ordered = list(umbrella_subjects.keys())

        # Execute ALL tools in parallel — regular tools plus master_paper_mcp
        # fan-outs. Master tasks never raise (errors are logged + returned as
        # empty lists), so index-based error attribution below stays correct
        # for the active_tools slice.
        tool_outputs = await asyncio.gather(
            *[_run_tool(t) for t in active_tools],
            *master_tasks,
            return_exceptions=True,
        )

        # Aggregate results
        for i, output in enumerate(tool_outputs):
            if isinstance(output, Exception):
                label = (
                    active_tools[i].tool_name
                    if i < len(active_tools)
                    else f"{MASTER_PAPER_MCP_NAME}:{master_umbrellas_ordered[i - len(active_tools)]}"
                )
                self._logger.warning(
                    "tool_execution_error",
                    tool=label,
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

    async def _analyze_cross_domain(
        self,
        query: str,
        deliberation: Deliberation,
        tool_results: dict[str, Any],
        progress: ProgressCallback = None,
    ) -> list[dict[str, Any]]:
        """Bridge AI methods ↔ Bio problems into candidate cross-applications.

        Runs a single LLM call that inspects the top AI evidence (arxiv
        papers, repos, HF models) alongside the top Bio evidence
        (pubmed_papers extracted from mcp_results, plus trials/compounds)
        and emits structured bridges:

        ``{ai_method, bio_target, mechanism, novelty, feasibility, testable_prediction}``

        Gating:
          * both ``ai_weight`` and ``bio_weight`` must be ≥ 0.1
          * need at least 1 AI item AND 1 bio signal (pubmed, trials, compounds)

        A parse failure returns [] rather than raising — bridges are an
        additive signal; the synthesis still runs without them.
        """
        weights = deliberation.intent_weights or {}
        if weights.get("ai", 0.0) < 0.1 or weights.get("bio", 0.0) < 0.1:
            return []

        papers = (tool_results.get("papers") or [])[:5]
        repos = (tool_results.get("repositories") or [])[:3]
        models = (tool_results.get("models") or [])[:3]
        pubmed_papers = self._extract_pubmed_papers(tool_results.get("mcp_results") or [])
        pubmed_top = pubmed_papers[:5]
        # master_paper_mcp papers bucketed by umbrella so they count as ai
        # or bio evidence for the bridge gate and the prompt blocks below.
        master_papers = self._extract_master_paper_papers(tool_results.get("mcp_results") or [])
        master_ai = [p for p in master_papers if p.get("umbrella") == "ai"][:3]
        master_bio = [p for p in master_papers if p.get("umbrella") == "bio"][:3]

        has_ai = bool(papers or repos or models or master_ai)
        has_bio = bool(pubmed_top) or bool(master_bio) or any(
            isinstance(r, dict) and r.get("source") in ("clinicaltrials-bio", "pubchem-bio", "doi-bio")
            for r in tool_results.get("mcp_results") or []
        )
        if not (has_ai and has_bio):
            return []

        def _fmt_paper(p: dict) -> str:
            title = p.get("title", "Untitled")
            url = p.get("url") or f"https://arxiv.org/abs/{p.get('arxiv_id', '')}"
            desc = (p.get("description") or p.get("summary") or "")[:200]
            return f"- {title} ({url}) — {desc}"

        def _fmt_repo(r: dict) -> str:
            return f"- {r.get('name') or r.get('full_name', '?')} ({r.get('url') or r.get('html_url', '')}): {(r.get('description') or '')[:150]}"

        def _fmt_model(m: dict) -> str:
            mid = m.get("id") or m.get("title", "?")
            return f"- {mid} (https://huggingface.co/{mid}): {(m.get('description') or '')[:120]}"

        ai_block = "\n".join(
            [_fmt_paper(p) for p in papers]
            + [_fmt_repo(r) for r in repos]
            + [_fmt_model(m) for m in models]
            + [_fmt_paper(p) for p in master_ai]
        ) or "(no AI evidence)"

        bio_block_parts: list[str] = []
        for p in pubmed_top:
            bio_block_parts.append(
                f"- {p.get('title', 'Untitled')} ({p['url']}) — {(p.get('description') or '')[:200]}"
            )
        for p in master_bio:
            bio_block_parts.append(_fmt_paper(p))
        # Pull a short trials/compounds summary from the raw mcp_results so the
        # bridge LLM has a chance to reason about them as bio targets.
        for r in tool_results.get("mcp_results") or []:
            if not isinstance(r, dict):
                continue
            src = r.get("source")
            if src in ("clinicaltrials-bio", "pubchem-bio", "doi-bio"):
                data = r.get("data")
                snippet = data[:400] if isinstance(data, str) else json.dumps(data, default=str)[:400]
                bio_block_parts.append(f"- [{src}] {snippet}")
        bio_block = "\n".join(bio_block_parts) or "(no Bio evidence)"

        prompt = f"""You are a cross-disciplinary analyst bridging AI methods and biomedical problems.

User query: {query}
Intent weights: AI={weights.get('ai', 0):.2f}, Bio={weights.get('bio', 0):.2f}, Web={weights.get('web', 0):.2f}

AI evidence (papers, repos, models):
{ai_block}

Bio evidence (literature, trials, compounds):
{bio_block}

Task: Identify 2–4 concrete BRIDGES where a specific AI method could be applied to a specific biomedical problem (or vice versa). Each bridge must reference real items from the evidence above — do NOT invent papers or tools.

Return a JSON array. Each element MUST have these keys exactly:
  "ai_method"       – short phrase + URL (from AI evidence)
  "bio_target"      – short phrase + URL (from Bio evidence)
  "mechanism"       – one sentence explaining why this bridge is plausible
  "novelty"         – float 0.0–1.0 (higher = less obvious pairing)
  "feasibility"     – float 0.0–1.0 (higher = easier to prototype)
  "testable_prediction" – one sentence describing a concrete experiment that could falsify the bridge

Return ONLY the JSON array. No prose. No markdown fences."""

        if progress:
            await progress("stage", {
                "stage": "bridging",
                "message": "Analyzing AI↔Bio cross-domain bridges...",
            })

        try:
            # 3000 tokens fits up to 4 bridges with full testable_prediction
            # sentences; 1200 was observed truncating the JSON array.
            raw = await self.think(
                prompt,
                task_type=TaskType.REASONING,
                temperature=0.4,
                max_tokens=3000,
            )
        except Exception as e:
            self._logger.warning("cross_domain_analysis_failed", error=str(e))
            return []

        # Strip code fences if the model added them despite instructions.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to salvage the first JSON array substring.
            m = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
            if not m:
                self._logger.warning("cross_domain_parse_failed", raw=raw[:200])
                return []
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                self._logger.warning("cross_domain_parse_failed", raw=raw[:200])
                return []

        if not isinstance(parsed, list):
            return []

        bridges: list[dict[str, Any]] = []
        for item in parsed[:4]:
            if not isinstance(item, dict):
                continue
            ai = item.get("ai_method") or item.get("ai")
            bio = item.get("bio_target") or item.get("bio")
            mech = item.get("mechanism") or item.get("why")
            if not (ai and bio and mech):
                continue
            bridges.append({
                "ai_method": str(ai),
                "bio_target": str(bio),
                "mechanism": str(mech),
                "novelty": float(item.get("novelty", 0.5) or 0.5),
                "feasibility": float(item.get("feasibility", 0.5) or 0.5),
                "testable_prediction": str(item.get("testable_prediction", "")),
            })

        self._logger.info(
            "cross_domain_bridges",
            count=len(bridges),
            ai_weight=weights.get("ai", 0),
            bio_weight=weights.get("bio", 0),
        )
        if progress and bridges:
            await progress("thinking", {
                "step_type": "reflection",
                "content": f"Identified {len(bridges)} cross-domain bridge(s) between AI methods and Bio problems.",
            })

        return bridges

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
            mcp_results = tool_results.get("mcp_results") or []
            master_papers = self._extract_master_paper_papers(mcp_results)
            pubmed_papers = self._extract_pubmed_papers(mcp_results)
            # Merge arxiv papers + master_paper_mcp papers so the debate sees
            # the full paper evidence surface, not just the arxiv slice.
            merged_papers = (
                (tool_results.get("papers") or [])
                + master_papers
                + pubmed_papers
            )
            ctx["papers"] = merged_papers[:10]
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

    # ---- master_paper_mcp integration --------------------------------
    # The master_paper_mcp HTTP gateway (registered in ~/.dova.json as
    # `master_paper_mcp`) is invoked additively for the ai / bio / web
    # umbrellas. We health-check it via `health_self` (short cache) and
    # silently skip the fan-out if it isn't up — keeping the gateway a
    # best-effort signal rather than a hard dependency.

    _MASTER_HEALTH_TTL_S: float = 30.0
    # Health probe must not block the critical path — if master_paper_mcp is
    # up but its health_self tool hangs, every request in the TTL window
    # would stall on it without this cap.
    _MASTER_HEALTH_PROBE_TIMEOUT_S: float = 3.0
    # Hard per-subject call timeout. The MCP client's default timeout is
    # 45s (~ `.dova.json`); a local wait_for bounds each shard independently
    # so a single stalled DB doesn't stretch the whole request.
    _MASTER_SUBJECT_TIMEOUT_S: float = 20.0
    # Hard per-tool wall-clock for top-level tool dispatch (arxiv/github/hf/
    # web/bio/…). Without it, one stalled tool gates the whole step's gather.
    _TOOL_EXEC_TIMEOUT_S: float = 30.0
    # Hard per-server timeout inside the bio umbrella fan-out. Bounds the
    # pubmed retry + hydration chain that can otherwise accumulate > 90s.
    _BIO_SERVER_TIMEOUT_S: float = 25.0
    # Per-subject failure memory TTL. A subject that errored/timed out
    # recently is skipped for this window to avoid repeated slow failures.
    _MASTER_SUBJECT_FAILURE_TTL_S: float = 120.0
    # Caps on fan-out width.
    _MASTER_MAX_SUBJECTS_PER_UMBRELLA: int = 2
    _MASTER_MAX_SUBJECTS_PER_REQUEST: int = 3

    def _select_master_subjects(
        self,
        umbrella: str,
        query: str,
        max_per_umbrella: int = 2,
    ) -> list[str]:
        """Rank umbrella subjects by query-keyword overlap.

        Returns at most ``max_per_umbrella`` subjects. When no keyword
        matches, falls back to the umbrella's default subject (one shard,
        not all of them) to keep the gateway useful without fan-out cost.
        """
        subjects = MASTER_PAPER_MCP_UMBRELLA_SUBJECTS.get(umbrella, [])
        if not subjects:
            return []
        q_low = (query or "").lower()
        scored: list[tuple[int, str]] = []
        for subj in subjects:
            kws = MASTER_PAPER_MCP_SUBJECT_KEYWORDS.get(subj, [])
            hits = sum(1 for kw in kws if kw and kw in q_low)
            if hits > 0:
                scored.append((hits, subj))
        if not scored:
            default = MASTER_PAPER_MCP_UMBRELLA_DEFAULT_SUBJECT.get(umbrella)
            return [default] if default else []
        scored.sort(key=lambda x: (-x[0], subjects.index(x[1])))
        return [s for _, s in scored[:max_per_umbrella]]

    async def _master_paper_mcp_healthy(self) -> bool:
        """Return True if master_paper_mcp's `health_self` tool responds.

        Result is cached for ``_MASTER_HEALTH_TTL_S`` to avoid probing on
        every search. Any exception or unsuccessful MCPToolResult counts
        as unhealthy.
        """
        cache: tuple[float, bool] | None = getattr(
            self, "_master_mcp_health_cache", None
        )
        now = time.time()
        if cache is not None and now - cache[0] < self._MASTER_HEALTH_TTL_S:
            return cache[1]
        healthy = False
        try:
            result = await asyncio.wait_for(
                self.call_tool(
                    MASTER_PAPER_MCP_NAME,
                    "health_self",
                    {},
                    cache_ttl=0,
                ),
                timeout=self._MASTER_HEALTH_PROBE_TIMEOUT_S,
            )
            healthy = bool(result.success)
            if not healthy:
                self._logger.info(
                    "master_paper_mcp_unhealthy",
                    error=result.error,
                )
        except asyncio.TimeoutError:
            self._logger.info(
                "master_paper_mcp_unhealthy",
                error=f"health_self timed out after {self._MASTER_HEALTH_PROBE_TIMEOUT_S}s",
            )
        except Exception as exc:
            self._logger.info(
                "master_paper_mcp_unhealthy",
                error=str(exc),
            )
        self._master_mcp_health_cache = (now, healthy)
        return healthy

    def _master_subject_recently_failed(self, subject: str) -> bool:
        """Return True if ``subject`` errored within the failure-TTL window."""
        cache: dict[str, float] = getattr(
            self, "_master_mcp_subject_failures", {}
        )
        ts = cache.get(subject)
        if ts is None:
            return False
        if time.time() - ts < self._MASTER_SUBJECT_FAILURE_TTL_S:
            return True
        cache.pop(subject, None)
        return False

    def _record_master_subject_failure(self, subject: str) -> None:
        cache: dict[str, float] = getattr(
            self, "_master_mcp_subject_failures", None
        ) or {}
        cache[subject] = time.time()
        self._master_mcp_subject_failures = cache

    async def _invoke_master_paper_mcp(
        self,
        umbrella: str,
        query: str,
        limit: int = 10,
        progress: ProgressCallback = None,
        subjects: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fan master_paper_mcp.search_papers over the selected subjects.

        ``subjects`` — pre-selected, query-relevance-ranked subject shards
        (see ``_select_master_subjects``). When omitted, falls back to the
        umbrella's full subject list so existing callers keep working.

        Each call is bounded by ``_MASTER_SUBJECT_TIMEOUT_S`` and subjects
        that errored/timed-out recently are skipped for
        ``_MASTER_SUBJECT_FAILURE_TTL_S`` to avoid repeated slow failures.
        Returns an empty list if the gateway is not healthy or the query
        is blank.
        """
        if subjects is None:
            subjects = MASTER_PAPER_MCP_UMBRELLA_SUBJECTS.get(umbrella, [])
        # Drop subjects that failed recently — they'd just time out again.
        subjects = [s for s in subjects if not self._master_subject_recently_failed(s)]
        if not subjects or not query.strip():
            return []
        if not await self._master_paper_mcp_healthy():
            self._logger.info(
                "master_paper_mcp_skipped",
                reason="unhealthy",
                umbrella=umbrella,
            )
            return []

        self._logger.info(
            "master_paper_mcp_fanout_starting",
            umbrella=umbrella,
            subjects=subjects,
            query=query,
        )
        if progress:
            await progress("stage", {
                "stage": "searching",
                "tool": f"{MASTER_PAPER_MCP_NAME}:{umbrella}",
                "message": f"Searching {MASTER_PAPER_MCP_NAME} ({umbrella})...",
            })

        async def _one(subject: str) -> dict[str, Any] | None:
            try:
                r = await asyncio.wait_for(
                    self.call_tool(
                        MASTER_PAPER_MCP_NAME,
                        "search_papers",
                        {"query": query, "subject": subject, "limit": limit},
                    ),
                    timeout=self._MASTER_SUBJECT_TIMEOUT_S,
                )
                if r.success and r.data:
                    return {
                        "source": f"{MASTER_PAPER_MCP_NAME}:{subject}",
                        "tool": "search_papers",
                        "data": r.data,
                    }
                if not r.success:
                    self._record_master_subject_failure(subject)
                    self._logger.warning(
                        "master_paper_mcp_subject_failed",
                        subject=subject,
                        error=r.error,
                    )
            except asyncio.TimeoutError:
                self._record_master_subject_failure(subject)
                self._logger.warning(
                    "master_paper_mcp_subject_timeout",
                    subject=subject,
                    timeout_s=self._MASTER_SUBJECT_TIMEOUT_S,
                )
            except Exception as exc:
                self._record_master_subject_failure(subject)
                self._logger.warning(
                    "master_paper_mcp_subject_exception",
                    subject=subject,
                    error=str(exc),
                )
            return None

        gathered = await asyncio.gather(*[_one(s) for s in subjects])
        return [r for r in gathered if r is not None]

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
                    # PubMed-specific progressive shortening: if the distilled
                    # query returned zero hits (Total Found: 0), drop the
                    # rightmost token and retry, down to 3 tokens. Long
                    # natural-language queries commonly zero-out under NCBI's
                    # boolean-AND parser even after stopword removal.
                    if (
                        name == "pubmed-bio"
                        and call == "pubmed_search_articles"
                        and r.success
                        and isinstance(r.data, str)
                        and "Total Found:** 0" in r.data
                    ):
                        tokens = params.get("query", "").split()
                        for n_keep in range(len(tokens) - 1, 2, -1):
                            short_query = " ".join(tokens[:n_keep])
                            retry_params = dict(params)
                            retry_params["query"] = short_query
                            self._logger.info(
                                "pubmed_retry_shorter",
                                original_tokens=len(tokens),
                                retry_tokens=n_keep,
                                short_query=short_query,
                            )
                            r2 = await self.call_tool(name, call, retry_params)
                            if (
                                r2.success
                                and isinstance(r2.data, str)
                                and "Total Found:** 0" not in r2.data
                            ):
                                r = r2
                                break
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

            async def _run_bounded(name: str) -> dict[str, Any] | None:
                try:
                    return await asyncio.wait_for(
                        _run(name), timeout=self._BIO_SERVER_TIMEOUT_S
                    )
                except asyncio.TimeoutError:
                    self._logger.warning(
                        "bio_server_timeout",
                        server=name,
                        timeout_s=self._BIO_SERVER_TIMEOUT_S,
                    )
                    return None

            fan_results = await asyncio.gather(*[_run_bounded(s) for s in selected])
            fan_results = [r for r in fan_results if r is not None]

            # Second pass: if the pubmed-bio search returned just a PMID list
            # (hosted MCP markdown format), hydrate it with titles + abstracts
            # via pubmed_fetch_articles so the synthesis LLM has real content.
            for r in fan_results:
                if r.get("source") != "pubmed-bio":
                    continue
                if r.get("tool") != "pubmed_search_articles":
                    continue
                data = r.get("data")
                if not isinstance(data, str) or "PMID" not in data:
                    continue
                pmids = re.findall(r"\b(\d{7,9})\b", data)
                # Dedup & cap to avoid giant fetches.
                unique_pmids = list(dict.fromkeys(pmids))[:10]
                if not unique_pmids:
                    continue
                try:
                    fetched = await self.call_tool(
                        "pubmed-bio",
                        "pubmed_fetch_articles",
                        {"pmids": unique_pmids},
                    )
                    if fetched.success and fetched.data:
                        r["data"] = fetched.data
                        r["tool"] = "pubmed_fetch_articles"
                        if isinstance(fetched.data, str):
                            data_size = f"{len(fetched.data)} chars"
                        else:
                            data_size = type(fetched.data).__name__
                        self._logger.info(
                            "pubmed_hydrated",
                            pmids=len(unique_pmids),
                            data=data_size,
                        )
                except Exception as e:
                    self._logger.warning("pubmed_hydrate_failed", error=str(e))

            return fan_results

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
            "doi-bio": [
                # DOI / cross-database citation signals — doi-mcp queries
                # CrossRef, OpenAlex, Semantic Scholar, DBLP, zbMATH, ERIC,
                # HAL, INSPIRE-HEP alongside PubMed. Distinct from pubmed-bio
                # because the user explicitly names the DB or asks to verify.
                "doi", "crossref", "openalex", "semantic scholar",
                "dblp", "zbmath", "inspirehep", "inspire-hep", "eric database",
                " hal ", "verify citation", "verified paper", "verified papers",
                "citation check", "citation verification", "anti-hallucination",
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

        # PubMed is literature — it's relevant to virtually any biomedical
        # question, so include it whenever the bio umbrella is invoked even
        # if the narrow keyword list didn't fire (e.g., "GLP-1 efficacy" hits
        # clinicaltrials but not pubmed). This prevents the selector from
        # silently starving the synthesis of literature context.
        if "pubmed-bio" in servers and "pubmed-bio" not in hits:
            hits.append("pubmed-bio")

        if hits:
            return hits

        # No servers matched and pubmed-bio not configured — fall back to the
        # first available.
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
            "doi-bio": [
                "doi", "crossref", "openalex", "semantic scholar",
                "dblp", "zbmath", "inspirehep", "inspire-hep",
                "verify citation", "verified paper", "verified papers",
                "citation check", "citation verification",
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
            "doi-bio": "findVerifiedPapers",
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
            # pubmed_search_articles: {query, maxResults, dateRange{minDate,maxDate}}
            # PubMed's parser treats the full string as a conjunction, so long
            # natural-language queries ("latest clinical trials for ... safety
            # profile and efficacy 2026") return zero hits. Distill to a short
            # keyword form: drop stopwords, years, recency/qualifier phrases.
            # Also constrain to the bio recency window (24 months). Date format
            # is PubMed's native YYYY/MM/DD.
            min_date = _months_ago_date(BIO_RECENCY_MONTHS, fmt="%Y/%m/%d")
            max_date = datetime.now().strftime("%Y/%m/%d")
            return {
                "query": _distill_pubmed_query(query),
                "maxResults": 10,
                "dateRange": {"minDate": min_date, "maxDate": max_date},
            }
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
        if server_name == "doi-bio":
            # findVerifiedPapers: {query, source, limit, yearFrom?, yearTo?}.
            # Use source="all" to fan out across all 9 DBs; bio recency
            # window keeps results aligned with the PubMed window.
            return {
                "query": query,
                "source": "all",
                "limit": 10,
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
        """Search ArXiv for papers within the AI recency window (12 months)."""
        date_from = _months_ago_date(AI_RECENCY_MONTHS, fmt="%Y-%m-%d")
        result = await self.search_arxiv(query, max_results=max_results, date_from=date_from)
        if not result.success:
            self._logger.warning("arxiv_search_failed", error=result.error, query=query, date_from=date_from)
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

        # Cross-domain AI↔Bio bridges produced by _analyze_cross_domain.
        # Injected as a distinct prompt section so the synthesis LLM weaves
        # bridges into the narrative instead of treating them as decoration.
        bridges = results.get("cross_domain_bridges") or []
        if bridges:
            bridge_lines = []
            for i, b in enumerate(bridges, 1):
                bridge_lines.append(
                    f"{i}. **{b.get('ai_method', '')}** ⇄ **{b.get('bio_target', '')}**\n"
                    f"   Mechanism: {b.get('mechanism', '')}\n"
                    f"   Novelty: {b.get('novelty', 0):.2f} · Feasibility: {b.get('feasibility', 0):.2f}\n"
                    f"   Testable prediction: {b.get('testable_prediction', '')}"
                )
            result_parts.append(
                "**Cross-Domain Bridges (AI ⇄ Bio):**\n" + "\n\n".join(bridge_lines)
            )

        # Drug-story chain (PubChem → PubMed → ClinicalTrials). Presented as
        # a coherent unit so the synthesis LLM treats compound, mechanism,
        # and evidence as one story instead of three disjoint mcp_results.
        story = results.get("drug_story")
        if story:
            story_lines = [f"- Compound: **{story.get('compound', '?')}**"]
            if story.get("pubchem_url"):
                story_lines.append(f"  PubChem: {story['pubchem_url']}")
            pmid_urls = story.get("mechanism_pmid_urls") or []
            if pmid_urls:
                story_lines.append(f"  Mechanism literature: {', '.join(pmid_urls)}")
            trial_urls = story.get("trial_urls") or []
            if trial_urls:
                story_lines.append(f"  Clinical trials: {', '.join(trial_urls)}")
            result_parts.append(
                "**Drug Story (PubChem → PubMed → ClinicalTrials):**\n"
                + "\n".join(story_lines)
            )

        if results.get("mcp_results"):
            # Truncate each bio source's payload proportionally to the bio
            # weight. A 10% bio query gets ~1500 chars per server; a 60% bio
            # query gets ~9000 chars so the hydrated PubMed / ClinicalTrials
            # markdown (titles + abstracts) is passed through mostly intact.
            bio_budget_chars = max(1500, int(12000 * weights.get("bio", 0.2)))
            for mcp_result in results["mcp_results"]:
                source = mcp_result.get("source", "unknown")
                data = mcp_result.get("data", {})
                if "pricing" in source.lower() and isinstance(data, (dict, list)):
                    fuzzy_result = self._process_pricing_with_fuzzy_match(query, data)
                    if fuzzy_result:
                        result_parts.append(fuzzy_result)
                        continue
                is_bio = source in ("pubmed-bio", "clinicaltrials-bio", "pubchem-bio", "doi-bio")
                # master_paper_mcp shards carry full paper records and deserve
                # the same synthesis budget as the bio MCPs — truncating them
                # to 500 chars was hiding entire paper payloads from the LLM.
                is_master_paper = source.startswith(f"{MASTER_PAPER_MCP_NAME}:")
                cap = bio_budget_chars if (is_bio or is_master_paper) else 500
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

        # Bio→AI reframe hints (Axis 1 #2). Injected when the query uses
        # biological vocabulary AND the AI group has meaningful weight, so
        # the synthesis draws structural analogies rather than just
        # describing bio findings in isolation. No LLM cost.
        bio_to_ai_block = ""
        if weights.get("ai", 0.0) >= 0.3:
            reframes = _select_bio_to_ai_reframes(query)
            if reframes:
                bio_to_ai_block = (
                    "\nBio→AI mechanism analogues to consider (when relevant to the question):\n"
                    + "\n".join(reframes)
                    + "\n"
                )

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
{inferred_context}{bio_to_ai_block}
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

6b. CROSS-DOMAIN BRIDGES (when the "Cross-Domain Bridges" section is present)
   - Weave at least 2 of the listed bridges into the narrative as a dedicated "Cross-Domain Opportunities" (or similarly named) section near the end of the answer.
   - For each bridge you use: cite BOTH the AI source URL AND the Bio source URL in the same paragraph, explain the mechanism in your own words, and state the testable prediction verbatim.
   - Prefer bridges with higher combined novelty + feasibility. Flag any bridge with novelty ≥ 0.7 as "speculative but high-reward".
   - Do NOT invent bridges not in the section; do NOT drop all bridges unless none of them actually bridge AI and Bio.

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
    def _extract_drug_story(
        mcp_results: list[dict],
        query: str,
    ) -> dict[str, Any] | None:
        """Chain PubChem → PubMed → ClinicalTrials into a coherent drug story.

        Pure string/dict processing — no LLM call. Extracts compound names
        from PubChem markdown, checks which compounds are mentioned in
        PubMed titles/abstracts and ClinicalTrials study lists, and emits a
        single structured summary:

            {compound, pubchem_url, mechanism_pmids, trial_nct_ids, summary}

        Returns None when there's not enough signal to form a story.
        """
        if not mcp_results:
            return None

        pubchem_blob = ""
        pubmed_blob = ""
        trials_blob = ""
        for r in mcp_results:
            if not isinstance(r, dict):
                continue
            data = r.get("data")
            if not isinstance(data, str):
                continue
            src = r.get("source")
            if src == "pubchem-bio":
                pubchem_blob += data + "\n"
            elif src == "pubmed-bio":
                pubmed_blob += data + "\n"
            elif src == "clinicaltrials-bio":
                trials_blob += data + "\n"

        if not pubchem_blob and not trials_blob:
            return None

        # PubChem markdown usually surfaces "CID: 12345" and a compound name.
        # Try to pull the top compound name and CID.
        cid_match = re.search(r"(?:CID|PubChem\s*CID)[:\s]+(\d+)", pubchem_blob, re.IGNORECASE)
        cid = cid_match.group(1) if cid_match else None
        # Compound name candidates: look for "Name: X" or a bolded header.
        name_match = re.search(r"(?:Name|IUPAC|Compound)[:\s]+([A-Za-z][\w\-,\(\) ]{2,80})", pubchem_blob)
        compound_name: str | None = name_match.group(1).strip() if name_match else None
        # Fallback: any capitalised token repeated across blobs could be the
        # subject — use the most prominent query token that looks like a
        # compound (hyphenated or ≥4 chars alpha) as a last resort.
        if not compound_name:
            candidates = re.findall(r"\b[A-Z][a-zA-Z\-]{3,20}\b", query)
            if candidates:
                compound_name = candidates[0]

        # Harvest PMIDs / NCT IDs that co-occur with the compound name in the
        # respective blobs. If we don't have a compound name, still return
        # the aggregate counts so the story block is useful.
        pmids = list(dict.fromkeys(re.findall(r"\b(\d{7,9})\b", pubmed_blob)))[:5]
        ncts = list(dict.fromkeys(re.findall(r"\bNCT\d{7,9}\b", trials_blob, re.IGNORECASE)))[:5]

        if not (compound_name or pmids or ncts):
            return None

        story = {
            "compound": compound_name or "(unknown)",
            "pubchem_cid": cid,
            "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}" if cid else None,
            "mechanism_pmids": pmids,
            "mechanism_pmid_urls": [f"https://pubmed.ncbi.nlm.nih.gov/{p}/" for p in pmids],
            "trial_nct_ids": ncts,
            "trial_urls": [f"https://clinicaltrials.gov/study/{n}" for n in ncts],
        }
        # Natural-language summary for inline use in answers.
        parts = [f"Drug story for **{story['compound']}**"]
        if cid:
            parts.append(f"(PubChem CID {cid}, {story['pubchem_url']})")
        if pmids:
            parts.append(f"literature: {len(pmids)} PubMed article(s) — {', '.join(pmids)}")
        if ncts:
            parts.append(f"trials: {len(ncts)} ClinicalTrials study(ies) — {', '.join(ncts)}")
        story["summary"] = "; ".join(parts) + "."
        return story

    @staticmethod
    def _extract_pubmed_papers(mcp_results: list[dict]) -> list[dict]:
        """Flatten PubMed MCP results into paper-shaped dicts.

        Tries list-of-dicts, {articles: [...]}/{results: [...]}, and finally
        a PMID regex sweep over any text payload so the caller always gets
        something usable with a URL field.

        The hosted pubmed.caseyjhand.com MCP returns a Markdown blob like
        ``**PMIDs:** 38843460, 39114288, ...``, so the string path also
        recognises that specific shape.
        """
        pmid_url_re = re.compile(r"(?:PMID[:\s]+|pubmed\.ncbi\.nlm\.nih\.gov/)(\d{5,9})", re.IGNORECASE)
        pmid_list_re = re.compile(r"(?i)\bPMIDs?\b\s*[:\*]*\s*([\d,\s]+?)(?:\n|$)")
        bare_pmid_re = re.compile(r"\b(\d{7,9})\b")  # fallback for `**PMIDs:** 38843460, ...`
        papers: list[dict] = []
        seen_pmids: set[str] = set()

        def _emit(pmid: str, title: str = "", description: str = "", authors: list | None = None) -> None:
            if not pmid or pmid in seen_pmids:
                return
            seen_pmids.add(pmid)
            papers.append({
                "title": title or f"PubMed PMID:{pmid}",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "description": description,
                "source": "pubmed",
                "metadata": {"pmid": pmid, "authors": authors or []},
            })

        def _walk(data: Any) -> None:
            if isinstance(data, list):
                for item in data:
                    _walk(item)
            elif isinstance(data, dict):
                pmid = str(data.get("pmid") or data.get("PMID") or data.get("uid") or "").strip()
                title = data.get("title") or data.get("Title") or data.get("articleTitle") or ""
                abstract = data.get("abstract") or data.get("Abstract") or data.get("description") or ""
                authors = data.get("authors") or data.get("Authors") or []
                if isinstance(authors, list):
                    author_names = [
                        a.get("name") if isinstance(a, dict) else str(a)
                        for a in authors
                    ]
                else:
                    author_names = []
                if pmid:
                    _emit(pmid, title, abstract[:400] if isinstance(abstract, str) else "", author_names)
                # Recurse into nested containers.
                for k in ("articles", "results", "data", "items", "PubmedArticle", "records"):
                    if k in data:
                        _walk(data[k])
            elif isinstance(data, str):
                # 1) Hydrated article blocks with titles + PMIDs.
                #    Shape: "### <title>\n...\n**PMID:** 12345"
                block_re = re.compile(
                    r"###\s+(?P<title>[^\n]+?)\s*\n(?P<body>.*?)(?=\n###\s|\Z)",
                    re.DOTALL,
                )
                for m in block_re.finditer(data):
                    title = m.group("title").strip()
                    body = m.group("body")
                    pmid_m = re.search(r"\*\*PMID:\*\*\s*(\d{5,9})", body)
                    if not pmid_m:
                        continue
                    abstract_m = re.search(r"####\s*Abstract\s*\n(.+?)(?=\n####|\n###|\Z)", body, re.DOTALL)
                    abstract = (abstract_m.group(1).strip() if abstract_m else "")[:400]
                    _emit(pmid_m.group(1), title=title, description=abstract)
                # 2) "PMID: 12345" or "pubmed.ncbi.nlm.nih.gov/12345" forms.
                for m in pmid_url_re.finditer(data):
                    _emit(m.group(1))
                # 3) Explicit PMID list line (hosted MCP markdown search format).
                for m in pmid_list_re.finditer(data):
                    for bm in bare_pmid_re.finditer(m.group(1)):
                        _emit(bm.group(1))

        for r in mcp_results:
            if not isinstance(r, dict):
                continue
            if r.get("source") != "pubmed-bio":
                continue
            _walk(r.get("data"))

        return papers

    @staticmethod
    def _extract_master_paper_papers(mcp_results: list[dict]) -> list[dict]:
        """Flatten master_paper_mcp.search_papers results into paper-shaped dicts.

        The gateway returns a list of Paper records
        ``{doi, title, authors, abstract, url, pdf_url, source, ...}`` per
        shard. We tag each paper with the umbrella (ai/bio/web) derived from
        the master_paper_mcp subject so downstream ranking/debate can bucket
        them correctly. Duplicates across subjects are removed by (doi | url
        | title).
        """
        # subject → umbrella mapping (matches MASTER_PAPER_MCP_UMBRELLA_SUBJECTS)
        subject_to_umbrella = {
            "ai": "ai", "computer": "ai", "math": "ai", "physics": "ai",
            "bio": "bio", "clinical": "bio", "chemistry": "bio",
            "social": "web", "other": "web",
        }
        papers: list[dict] = []
        seen: set[str] = set()

        for r in mcp_results:
            if not isinstance(r, dict):
                continue
            source = r.get("source") or ""
            if not source.startswith(f"{MASTER_PAPER_MCP_NAME}:"):
                continue
            subject = source.split(":", 1)[1] if ":" in source else ""
            umbrella = subject_to_umbrella.get(subject, "web")
            data = r.get("data")
            # master_paper_mcp returns list[dict]; a stringified payload would
            # be a downstream bug — skip rather than regex-scrape.
            if not isinstance(data, list):
                continue
            for item in data:
                if not isinstance(item, dict):
                    continue
                title = (item.get("title") or "").strip()
                url = (item.get("url") or item.get("pdf_url") or "").strip()
                doi = (item.get("doi") or "").strip()
                key = doi.lower() or url.lower() or title.lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                authors = item.get("authors") or []
                if not isinstance(authors, list):
                    authors = []
                abstract = (item.get("abstract") or "")[:500]
                papers.append({
                    "title": title or "Untitled",
                    "url": url,
                    "description": abstract,
                    "authors": authors,
                    "doi": doi,
                    "source": f"{MASTER_PAPER_MCP_NAME}:{subject}" if subject else MASTER_PAPER_MCP_NAME,
                    "umbrella": umbrella,
                    "published": item.get("published_date") or item.get("updated_date") or "",
                })
        return papers

    @staticmethod
    def _allocate_top_papers(
        arxiv_papers: list[dict],
        pubmed_papers: list[dict],
        intent_weights: dict[str, float] | None,
        cap: int = 10,
    ) -> list[dict]:
        """Pick up to `cap` papers, split between arxiv (ai weight) and pubmed (bio weight).

        Honors ``intent_weights`` proportionally. When one pool is empty, we
        do NOT silently pour the other pool into its slots — this would hide
        the fact that the dominant group failed. Instead:

        * If the empty pool is the MINORITY (weight < 0.3) we backfill from
          the majority pool, since the UI's dominant section is healthy.
        * If the empty pool is the MAJORITY (weight >= 0.3) we return only
          the minority's fair share and log the miss, so the UI honestly
          shows a short list rather than a misleading full page of minority
          results that contradict the deliberation banner.
        """
        weights = intent_weights or {}
        ai_w = float(weights.get("ai", 0.0))
        bio_w = float(weights.get("bio", 0.0))

        if not arxiv_papers and not pubmed_papers:
            return []

        total = ai_w + bio_w
        if total <= 0:
            ai_share, bio_share = 0.7, 0.3
        else:
            ai_share, bio_share = ai_w / total, bio_w / total

        MAJORITY_THRESHOLD = 0.30

        # Empty-pool handling with majority-respect guard.
        if not pubmed_papers and arxiv_papers:
            if bio_share >= MAJORITY_THRESHOLD:
                # Bio was the user's dominant intent but returned nothing;
                # return only the AI fair-share so we don't mislead.
                logger.warning(
                    "top_papers_bio_empty_majority",
                    bio_share=round(bio_share, 2),
                    ai_share=round(ai_share, 2),
                )
                return arxiv_papers[:max(1, round(ai_share * cap))]
            return arxiv_papers[:cap]
        if not arxiv_papers and pubmed_papers:
            if ai_share >= MAJORITY_THRESHOLD:
                logger.warning(
                    "top_papers_arxiv_empty_majority",
                    ai_share=round(ai_share, 2),
                    bio_share=round(bio_share, 2),
                )
                return pubmed_papers[:max(1, round(bio_share * cap))]
            return pubmed_papers[:cap]

        ai_slots = round(ai_share * cap)
        bio_slots = cap - ai_slots
        # Guarantee at least 1 slot per non-empty pool only when that side
        # has any weight — respect a hard-0 weight (e.g. bio=1.0 query).
        if ai_slots == 0 and ai_share > 0:
            ai_slots, bio_slots = 1, cap - 1
        elif bio_slots == 0 and bio_share > 0:
            bio_slots, ai_slots = 1, cap - 1

        ai_cut = arxiv_papers[:ai_slots]
        bio_cut = pubmed_papers[:bio_slots]
        # If one pool underfilled its slots, give the leftover to the other.
        leftover = cap - (len(ai_cut) + len(bio_cut))
        if leftover > 0:
            if len(ai_cut) < ai_slots:
                bio_cut = pubmed_papers[:bio_slots + leftover]
            else:
                ai_cut = arxiv_papers[:ai_slots + leftover]

        return ai_cut + bio_cut

    @staticmethod
    def extract_research_data(result_data: dict) -> dict:
        """Map orchestrator output to flat research arrays for ResearchResponse.

        Adds:
          - `pubmed_papers`: PubMed results flattened to paper-shaped dicts.
          - `top_papers`: up to 10 paper URLs, split between arxiv and pubmed
            proportionally to `deliberation.intent_weights` (ai vs bio).
        """
        action_result = result_data.get("action_result") or {}
        deliberation = result_data.get("deliberation", {}) or {}
        arxiv_papers = action_result.get("papers", []) or []
        mcp_results = action_result.get("mcp_results", []) or []
        pubmed_papers = ThinkingOrchestrator._extract_pubmed_papers(mcp_results)
        # master_paper_mcp papers bucket into the ai or bio pool by umbrella
        # so they flow through the same top_papers allocator. Tagged papers
        # are also surfaced as a dedicated `master_paper_papers` array for
        # clients that want to render the source explicitly.
        master_paper_papers = ThinkingOrchestrator._extract_master_paper_papers(mcp_results)
        master_ai = [p for p in master_paper_papers if p.get("umbrella") == "ai"]
        master_bio = [p for p in master_paper_papers if p.get("umbrella") == "bio"]
        top_papers = ThinkingOrchestrator._allocate_top_papers(
            arxiv_papers + master_ai,
            pubmed_papers + master_bio,
            deliberation.get("intent_weights"),
            cap=10,
        )
        return {
            "response": result_data.get("response", ""),
            "papers": arxiv_papers,
            "pubmed_papers": pubmed_papers,
            "master_paper_papers": master_paper_papers,
            "top_papers": top_papers,
            "cross_domain_bridges": action_result.get("cross_domain_bridges", []),
            "drug_story": action_result.get("drug_story"),
            "repositories": action_result.get("repositories", []),
            "models": action_result.get("models", []),
            "datasets": action_result.get("datasets", []),
            "web_results": action_result.get("web_results", []),
            "images": action_result.get("images", []),
            "mcp_results": mcp_results,
            "deliberation": deliberation,
            # Budget / per-stage telemetry (paper's Thm.1 + per-stage counts)
            "token_budget_estimate": action_result.get("token_budget_estimate"),
            "stage_tokens": action_result.get("stage_tokens", {}),
            "stage_tokens_total": action_result.get("stage_tokens_total", 0),
        }

    def register_agent(self, agent_type: str, agent: BaseAgent) -> None:
        """Register a specialized agent."""
        self.agents[agent_type] = agent
        self._logger.info("agent_registered", agent_type=agent_type)
