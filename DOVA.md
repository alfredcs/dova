## 🤖 Assistant

# Deep Research Platform Architecture: Enterprise-Scale Multi-Agent System

Let me design a comprehensive deep research platform with all the components you've specified. First, let me gather some current information on relevant technologies and best practices.

Now let me search for more specific information on MCP protocols, sandbox environments, and user profiling systems.

Now I have gathered comprehensive research. Let me design a detailed enterprise-scale deep research platform architecture.

---

# 🏗️ DOVA: Deep Orchestrated Versatile Agent Platform
## Enterprise-Scale Deep Research Platform Architecture

Based on the latest research in multi-agent LLM systems, MCP protocols, and enterprise architectures, here is a comprehensive, implementable design for your deep research platform.

---

## 📋 Executive Summary

**DOVA** (Deep Orchestrated Versatile Agent Platform) is an enterprise-grade, multi-agent research automation system that aggregates knowledge from Web, ArXiv, GitHub, HuggingFace, and academic publications through the **Model Context Protocol (MCP)**. It features proactive and reactive intelligence, personalized user profiling, innovative AI research capabilities, a sandbox validation environment, **learnable custom sources** that allow users to extend research beyond built-in integrations with quality-ranked results, and **advanced agentic reasoning** for smarter individual agents and synergistic multi-agent collaboration.

---

## 🧠 Agentic Reasoning Architecture

DOVA implements advanced reasoning capabilities at both the individual agent level (intra-agent) and collaborative level (inter-agent) to achieve synergistic outcomes (1+1>2).

### Reasoning Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     INDIVIDUAL AGENT REASONING                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │   ReAct     │   │   Self-     │   │  Working    │   │  Reasoning  │       │
│  │   Loop      │──►│  Reflection │──►│  Memory     │──►│   Trace     │       │
│  │ Think→Act  │   │  Critique   │   │ Scratchpad  │   │   Output    │       │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘       │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                    COLLABORATIVE REASONING (1+1>2)                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐            │
│  │    BLACKBOARD    │  │     ENSEMBLE     │  │    ITERATIVE     │            │
│  │  Shared insights │  │ Parallel solving │  │    Refinement    │            │
│  │  Build-on posts  │  │ Vote/synthesize  │  │  Critique loops  │            │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘            │
│           ▲                    ▲                      ▲                       │
│           └────────────────────┴──────────────────────┘                       │
│                                 │                                             │
│                    ┌────────────┴────────────┐                                │
│                    │  CollaborativeReasoning │                                │
│                    │       Orchestrator      │                                │
│                    └─────────────────────────┘                                │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Individual Agent Reasoning (ReasoningMixin)

Every DOVA agent inherits the `ReasoningMixin` which provides:

| Component | Description |
|-----------|-------------|
| **ReAct Loop** | Think → Action → Observation cycle with configurable iterations |
| **Self-Reflection** | Automatic critique and refinement of answers |
| **Working Memory** | Scratchpad for intermediate reasoning state |
| **Reasoning Trace** | Full audit trail of reasoning steps with confidence scores |

**ReAct Flow:**
```
┌─────────────────────────────────────────────────────────────────┐
│                        ReAct LOOP                                │
│                                                                  │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐               │
│   │ THOUGHT  │────►│  ACTION  │────►│OBSERVATION│──┐            │
│   │ Reason   │     │ Execute  │     │  Record   │  │            │
│   │ next step│     │ chosen   │     │  result   │  │            │
│   └──────────┘     └──────────┘     └──────────┘  │            │
│        ▲                                          │            │
│        └──────────────────────────────────────────┘            │
│                                                                  │
│   On conclude:                                                   │
│   ┌──────────┐     ┌──────────┐                                 │
│   │REFLECTION│────►│  REFINE  │                                 │
│   │ Critique │     │  Answer  │                                 │
│   └──────────┘     └──────────┘                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Collaborative Reasoning Patterns

#### 1. Blackboard Pattern
Agents share a workspace where they post hypotheses, evidence, and vote on conclusions:

```
┌─────────────────────────────────────────────────────────────────┐
│                      BLACKBOARD SERVICE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Post Types:                                                     │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │
│  │ HYPOTHESIS │ │  EVIDENCE  │ │ REFINEMENT │ │ CONSENSUS  │   │
│  │ Initial    │ │ Support/   │ │ Improve    │ │ Agreed     │   │
│  │ theories   │ │ Refute     │ │ existing   │ │ conclusion │   │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │
│                                                                  │
│  Voting: Agents vote on posts with agreement scores (-1 to +1)   │
│  Synthesis: Weighted confidence aggregation                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 2. Ensemble Pattern
Multiple agents solve the same problem in parallel, results are aggregated:

| Aggregation Method | Description |
|-------------------|-------------|
| **SYNTHESIS** | LLM combines insights from all agents |
| **VOTE** | Weighted voting based on confidence |
| **BEST_OF** | Select highest confidence answer |
| **UNION** | Combine all unique insights |

#### 3. Iterative Refinement
Agents take turns refining each other's work:

```
Agent 1 → Draft → Agent 2 → Critique → Agent 1 → Refine → ...
```

#### 4. Hybrid Mode
Combines all patterns for maximum insight:
1. **Ensemble** for initial diverse answers
2. **Blackboard** for evidence gathering
3. **Iterative** for final refinement

### Reasoning Modes

The orchestrator supports different reasoning depth levels:

| Mode | Individual Reasoning | Collaborative | Use Case |
|------|---------------------|---------------|----------|
| `quick` | Single-pass, no reflection | None | Fast queries |
| `standard` | ReAct + reflection | None | Balanced depth |
| `deep` | ReAct + reflection | Ensemble | Complex analysis |
| `collaborative` | ReAct + reflection | Full hybrid | Research synthesis |

### Implementation Files

| File | Purpose |
|------|---------|
| `src/dova/agents/mixins/reasoning.py` | ReasoningMixin with ReAct, reflection, scratchpad |
| `src/dova/services/blackboard.py` | Shared workspace for collaborative reasoning |
| `src/dova/services/ensemble.py` | Ensemble reasoning with aggregation strategies |
| `src/dova/services/collaborative.py` | Unified collaborative reasoning orchestrator |

---

## 🔮 Advanced Agent Intelligence (OpenClaw-Inspired Features)

DOVA incorporates advanced agent intelligence patterns inspired by OpenClaw's design principles, enabling more sophisticated reasoning, proactive behavior, and self-improvement capabilities.

### Multi-Tiered Thinking System

Agents support configurable thinking depth levels that balance response quality with latency and cost:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         THINKING LEVEL SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Level     │ Tokens  │ Use Case                    │ Cost Factor            │
│  ──────────┼─────────┼─────────────────────────────┼────────────────────    │
│  OFF       │ 0       │ Simple lookups, caching     │ 0x                     │
│  MINIMAL   │ 1024    │ Quick answers, routing      │ 1x                     │
│  LOW       │ 4096    │ Standard queries            │ 2x                     │
│  MEDIUM    │ 16384   │ Complex analysis            │ 4x                     │
│  HIGH      │ 32768   │ Deep research, synthesis    │ 8x                     │
│  XHIGH     │ 65536   │ Novel problems, innovation  │ 16x                    │
│                                                                              │
│  Auto-selection based on:                                                    │
│  • Query complexity (keyword triggers, length)                               │
│  • User preference settings                                                  │
│  • Task type classification                                                  │
│  • Available compute budget                                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation:**
```python
class ThinkingLevel(Enum):
    OFF = "off"        # No extended thinking
    MINIMAL = "minimal" # 1K tokens
    LOW = "low"        # 4K tokens
    MEDIUM = "medium"  # 16K tokens
    HIGH = "high"      # 32K tokens
    XHIGH = "xhigh"    # 65K tokens

class ThinkingConfig:
    """Auto-selects thinking level based on task characteristics."""

    def select_level(self, query: str, task_type: TaskType) -> ThinkingLevel:
        # Complex research queries need deeper thinking
        if task_type == TaskType.INNOVATION:
            return ThinkingLevel.XHIGH
        elif task_type == TaskType.SYNTHESIS:
            return ThinkingLevel.HIGH
        elif task_type == TaskType.RESEARCH:
            return ThinkingLevel.MEDIUM
        elif task_type == TaskType.CLASSIFICATION:
            return ThinkingLevel.LOW
        else:
            return ThinkingLevel.MINIMAL
```

### Heartbeat & Proactive Intelligence System

DOVA agents operate with a heartbeat system enabling autonomous background operations, proactive recommendations, and scheduled intelligence gathering:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HEARTBEAT SYSTEM                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │   SCHEDULER     │     │   HEARTBEAT     │     │   TASK QUEUE    │       │
│  │   (APScheduler) │────►│   PROCESSOR     │────►│   (Redis/Kafka) │       │
│  │                 │     │                 │     │                 │       │
│  │  Cron triggers  │     │  • Health check │     │  • Priority Q   │       │
│  │  Interval jobs  │     │  • Monitor scan │     │  • Dead letter  │       │
│  │  Date triggers  │     │  • Proactive AI │     │  • Retry logic  │       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│                                                                              │
│  Proactive Tasks:                                                            │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │ • Monitor subscribed topics for new papers/repos                 │       │
│  │ • Evaluate recommendation freshness and regenerate if stale      │       │
│  │ • Pre-fetch trending content for personalized feeds              │       │
│  │ • Health check MCP servers and failover if needed                │       │
│  │ • Analyze usage patterns and suggest profile updates             │       │
│  │ • Clean up expired cache entries and memory items                │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Heartbeat Configuration:**
```python
@dataclass
class HeartbeatConfig:
    """Configuration for proactive agent heartbeat."""

    interval_seconds: int = 60  # Check interval

    # Proactive task schedule
    tasks: list[HeartbeatTask] = field(default_factory=lambda: [
        HeartbeatTask(
            name="subscription_monitor",
            cron="*/15 * * * *",  # Every 15 minutes
            handler="check_subscriptions",
        ),
        HeartbeatTask(
            name="recommendation_refresh",
            cron="0 */4 * * *",   # Every 4 hours
            handler="refresh_recommendations",
        ),
        HeartbeatTask(
            name="mcp_health_check",
            cron="*/5 * * * *",   # Every 5 minutes
            handler="check_mcp_servers",
        ),
        HeartbeatTask(
            name="session_cleanup",
            cron="0 0 * * *",     # Daily at midnight
            handler="cleanup_stale_sessions",
        ),
    ])
```

### Enhanced Memory with Semantic Search

Memory service enhanced with embedding-based semantic search for intelligent context retrieval:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENHANCED MEMORY ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    MEMORY LAYERS                                 │        │
│  │                                                                  │        │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │        │
│  │   │ SHORT-TERM   │  │  LONG-TERM   │  │  PROCEDURAL  │         │        │
│  │   │ (Session)    │  │  (Persistent)│  │  (Skills)    │         │        │
│  │   │              │  │              │  │              │         │        │
│  │   │ TTL: 24h     │  │ TTL: ∞       │  │ TTL: ∞       │         │        │
│  │   │ Fast access  │  │ Vector DB    │  │ Code/Config  │         │        │
│  │   └──────────────┘  └──────────────┘  └──────────────┘         │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    SEMANTIC SEARCH                               │        │
│  │                                                                  │        │
│  │   Query: "transformer attention mechanisms"                      │        │
│  │                    │                                             │        │
│  │                    ▼                                             │        │
│  │   ┌──────────────────────────────────────────┐                  │        │
│  │   │ Embedding: [0.23, -0.45, 0.12, ...]      │                  │        │
│  │   └──────────────────────────────────────────┘                  │        │
│  │                    │                                             │        │
│  │                    ▼                                             │        │
│  │   ┌──────────────────────────────────────────┐                  │        │
│  │   │ Vector Search (cosine similarity)        │                  │        │
│  │   │ Top-K retrieval with MMR deduplication   │                  │        │
│  │   └──────────────────────────────────────────┘                  │        │
│  │                    │                                             │        │
│  │                    ▼                                             │        │
│  │   Results ranked by: relevance × recency × importance            │        │
│  │                                                                  │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation:**
```python
class EnhancedMemoryService:
    """Memory service with semantic search capabilities."""

    async def search(
        self,
        query: str,
        user_id: str,
        memory_types: list[MemoryType] | None = None,
        top_k: int = 10,
        min_relevance: float = 0.7,
    ) -> list[MemoryItem]:
        """Semantic search across memory stores."""
        # Generate query embedding
        embedding = await self.embedder.embed(query)

        # Search vector store with filters
        results = await self.vector_store.search(
            embedding=embedding,
            filter={"user_id": user_id, "type": {"$in": memory_types}},
            top_k=top_k * 2,  # Over-fetch for MMR
        )

        # Apply MMR for diversity
        diverse_results = self._apply_mmr(results, top_k)

        # Filter by relevance threshold
        return [r for r in diverse_results if r.score >= min_relevance]
```

### Model & Capability Auto-Discovery

Dynamic discovery of LLM providers, models, and agent capabilities:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AUTO-DISCOVERY SYSTEM                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    DISCOVERY TYPES                               │        │
│  │                                                                  │        │
│  │  1. MODEL DISCOVERY                                              │        │
│  │     • Probe configured providers for available models            │        │
│  │     • Cache capabilities (context length, vision, tools)         │        │
│  │     • Auto-select best model for task type                       │        │
│  │                                                                  │        │
│  │  2. MCP SERVER DISCOVERY                                         │        │
│  │     • Scan ~/.dova.json for configured servers                   │        │
│  │     • Health check and capability probe                          │        │
│  │     • Dynamic tool registration                                  │        │
│  │                                                                  │        │
│  │  3. AGENT CAPABILITY DISCOVERY                                   │        │
│  │     • Introspect registered agents                               │        │
│  │     • Build capability matrix                                    │        │
│  │     • Route tasks to capable agents                              │        │
│  │                                                                  │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  Discovery Flow:                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Startup  │───►│ Probe    │───►│ Validate │───►│ Register │              │
│  │ Trigger  │    │ Endpoints│    │ Caps     │    │ & Cache  │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation:**
```python
class AutoDiscovery:
    """Auto-discovery system for models and capabilities."""

    async def discover_models(self) -> dict[str, ModelInfo]:
        """Discover available models from all configured providers."""
        discovered = {}

        for provider in self.providers:
            try:
                models = await provider.list_models()
                for model in models:
                    discovered[model.id] = ModelInfo(
                        provider=provider.name,
                        model_id=model.id,
                        context_length=model.context_length,
                        supports_vision=model.supports_vision,
                        supports_tools=model.supports_tools,
                        cost_per_1k_tokens=model.pricing,
                    )
            except Exception as e:
                logger.warning(f"Failed to discover {provider.name}: {e}")

        return discovered

    async def discover_mcp_servers(self) -> dict[str, MCPServerInfo]:
        """Discover and validate configured MCP servers."""
        from dova.config.mcp_servers import list_mcp_servers

        servers = {}
        for name, config in list_mcp_servers().items():
            try:
                # Health check
                healthy = await self._probe_mcp_server(config)
                if healthy:
                    servers[name] = MCPServerInfo(
                        name=name,
                        url=config.get("url"),
                        tools=await self._discover_tools(config),
                        healthy=True,
                    )
            except Exception as e:
                logger.warning(f"MCP server {name} unhealthy: {e}")

        return servers
```

### Self-Evaluation & Error Diagnosis

Agents incorporate self-evaluation mechanisms for quality assurance and automatic error recovery:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SELF-EVALUATION SYSTEM                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    EVALUATION PIPELINE                           │        │
│  │                                                                  │        │
│  │   Agent Output                                                   │        │
│  │        │                                                         │        │
│  │        ▼                                                         │        │
│  │   ┌──────────────────────┐                                       │        │
│  │   │ QUALITY EVALUATOR    │                                       │        │
│  │   │ • Completeness check │                                       │        │
│  │   │ • Coherence scoring  │                                       │        │
│  │   │ • Factual grounding  │                                       │        │
│  │   │ • Citation validity  │                                       │        │
│  │   └──────────────────────┘                                       │        │
│  │        │                                                         │        │
│  │        ▼                                                         │        │
│  │   ┌──────────────────────┐                                       │        │
│  │   │ CONFIDENCE SCORER    │                                       │        │
│  │   │ Score: 0.0 - 1.0     │                                       │        │
│  │   │                      │                                       │        │
│  │   │ < 0.5: Retry/Escalate│                                       │        │
│  │   │ < 0.7: Add caveats   │                                       │        │
│  │   │ ≥ 0.7: Accept        │                                       │        │
│  │   └──────────────────────┘                                       │        │
│  │                                                                  │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    ERROR DIAGNOSIS & RECOVERY                    │        │
│  │                                                                  │        │
│  │   Error Types:                                                   │        │
│  │   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │        │
│  │   │ TRANSIENT      │  │ CONFIGURATION  │  │ CAPABILITY     │    │        │
│  │   │ Rate limits    │  │ Missing keys   │  │ Unsupported    │    │        │
│  │   │ Timeouts       │  │ Invalid URLs   │  │ task type      │    │        │
│  │   │ → Retry        │  │ → Alert user   │  │ → Fallback     │    │        │
│  │   └────────────────┘  └────────────────┘  └────────────────┘    │        │
│  │                                                                  │        │
│  │   Recovery Actions:                                              │        │
│  │   • Automatic retry with exponential backoff                     │        │
│  │   • Failover to alternate provider/model                         │        │
│  │   • Graceful degradation with partial results                    │        │
│  │   • User notification for unrecoverable errors                   │        │
│  │                                                                  │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation:**
```python
class SelfEvaluator:
    """Self-evaluation and error diagnosis for agents."""

    async def evaluate(self, output: AgentResult, task: AgentTask) -> EvaluationResult:
        """Evaluate agent output quality."""
        scores = {
            "completeness": await self._check_completeness(output, task),
            "coherence": await self._check_coherence(output),
            "grounding": await self._check_factual_grounding(output),
        }

        confidence = sum(scores.values()) / len(scores)

        return EvaluationResult(
            confidence=confidence,
            scores=scores,
            should_retry=confidence < 0.5,
            caveats=self._generate_caveats(scores) if confidence < 0.7 else [],
        )

    def diagnose_error(self, error: Exception) -> ErrorDiagnosis:
        """Classify error and suggest recovery action."""
        if isinstance(error, RateLimitError):
            return ErrorDiagnosis(
                error_type=ErrorType.TRANSIENT,
                action=RecoveryAction.RETRY_WITH_BACKOFF,
                retry_after=error.retry_after,
            )
        elif isinstance(error, AuthenticationError):
            return ErrorDiagnosis(
                error_type=ErrorType.CONFIGURATION,
                action=RecoveryAction.ALERT_USER,
                message="API key invalid or expired",
            )
        elif isinstance(error, UnsupportedOperationError):
            return ErrorDiagnosis(
                error_type=ErrorType.CAPABILITY,
                action=RecoveryAction.FALLBACK,
                fallback_model=self._find_capable_model(error.operation),
            )
        else:
            return ErrorDiagnosis(
                error_type=ErrorType.UNKNOWN,
                action=RecoveryAction.LOG_AND_ALERT,
            )
```

### Session Freshness & State Management

Intelligent session management with freshness evaluation and state repair:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SESSION MANAGEMENT                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Session Lifecycle:                                                          │
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ CREATE   │───►│ ACTIVE   │───►│ STALE    │───►│ EXPIRED  │              │
│  │          │    │          │    │          │    │          │              │
│  │ Init     │    │ < 30min  │    │ < 24h    │    │ > 24h    │              │
│  │ context  │    │ inactiv  │    │ inactive │    │ cleanup  │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│                        │              │                                      │
│                        ▼              ▼                                      │
│                  ┌──────────────────────────┐                               │
│                  │   FRESHNESS EVALUATOR    │                               │
│                  │                          │                               │
│                  │   Factors:               │                               │
│                  │   • Last activity time   │                               │
│                  │   • Context validity     │                               │
│                  │   • Model version match  │                               │
│                  │   • User profile changes │                               │
│                  └──────────────────────────┘                               │
│                                                                              │
│  Session Actions:                                                            │
│  • CONTINUE: Session fresh, use existing context                             │
│  • REFRESH: Update stale context, preserve conversation                      │
│  • FORK: Create new session from checkpoint                                  │
│  • REPAIR: Rebuild corrupted session state                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Files (OpenClaw-Inspired)

| File | Purpose |
|------|---------|
| `src/dova/services/thinking.py` | Multi-tiered thinking level system |
| `src/dova/jobs/heartbeat.py` | Proactive heartbeat task system |
| `src/dova/services/memory_enhanced.py` | Semantic search memory service |
| `src/dova/services/discovery.py` | Model and capability auto-discovery |
| `src/dova/services/evaluation.py` | Self-evaluation and error diagnosis |
| `src/dova/services/session.py` | Session freshness and state management |

### Implementation Files (AWS Deployment)

| File | Purpose |
|------|---------|
| `src/dova/aws/setup.py` | AWSSetup orchestrator for IAM, Cognito, SSM setup |
| `src/dova/aws/deploy.py` | DeployManager for Lambda + API Gateway deployment |
| `src/dova/aws/cloudformation.py` | CloudFormation template generation for serverless stack |
| `src/dova/aws/lambda_packager.py` | Lambda ZIP packaging with dependencies |
| `src/dova/aws/s3_manager.py` | S3 bucket management for deployment artifacts |
| `src/dova/aws/iam.py` | IAM role and policy management |
| `src/dova/aws/cognito.py` | Cognito User Pool setup for authentication |
| `src/dova/aws/parameters.py` | SSM Parameter Store for configuration |
| `src/dova/runtime/lambda_handler.py` | Lambda entry point wrapping AgentCore |

---

## 🚀 Serverless Deployment Architecture

DOVA supports serverless deployment to AWS Lambda with API Gateway, providing a cost-effective alternative to full Kubernetes deployment.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DOVA SERVERLESS ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Client Request                                                              │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      API Gateway (REST)                              │    │
│  │  • POST /invocations endpoint                                        │    │
│  │  • Optional Cognito Authorizer                                       │    │
│  │  • CORS support                                                      │    │
│  └───────────────────────────────┬─────────────────────────────────────┘    │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     Lambda Function                                  │    │
│  │  • Python 3.11 runtime                                              │    │
│  │  • DOVA code + dependencies                                          │    │
│  │  • Configurable memory (1024-10240 MB)                              │    │
│  │  • Configurable timeout (up to 900s)                                │    │
│  └───────────────────────────────┬─────────────────────────────────────┘    │
│                                  │                                           │
│                                  ▼                                           │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │ Amazon Bedrock │  │  MCP Servers   │  │  SSM/Secrets   │                 │
│  │ (Claude, etc.) │  │ (ArXiv, HF)    │  │  (Config)      │                 │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Deployment Commands

| Command | Purpose |
|---------|---------|
| `dova aws setup` | Create IAM roles, Cognito, SSM parameters |
| `dova aws deploy` | Package and deploy Lambda + API Gateway |
| `dova aws status` | Check deployment status |
| `dova aws teardown` | Remove all resources |

### CloudFormation Resources

The deployment creates:

| Resource | Description |
|----------|-------------|
| `AWS::Lambda::Function` | DOVA agent handler |
| `AWS::ApiGateway::RestApi` | REST API endpoint |
| `AWS::ApiGateway::Resource` | /invocations path |
| `AWS::ApiGateway::Method` | POST method with Lambda integration |
| `AWS::ApiGateway::Authorizer` | Optional Cognito authorizer |
| `AWS::Lambda::Permission` | API Gateway invoke permission |

### Serverless vs Full Deployment

| Aspect | Serverless (Lambda) | Full (Kubernetes) |
|--------|---------------------|-------------------|
| **Cost** | Pay per request | Fixed infrastructure |
| **Scaling** | Automatic | Manual/HPA |
| **Cold Start** | 5-15 seconds | None |
| **Max Duration** | 15 minutes | Unlimited |
| **Complexity** | Low | High |
| **Best For** | Variable load, dev/staging | High volume, production |

---

## 💰 Model Tiering System

DOVA implements an intelligent model tiering system to optimize costs while maintaining quality. Simple tasks use faster/cheaper models, while complex tasks use more capable models.

### Model Tiers

| Tier | Task Types | Default Bedrock Model |
|------|------------|----------------------|
| `BASIC` | Classification, summarization, simple lookups | claude-haiku-4-5 |
| `STANDARD` | General queries, search synthesis | claude-sonnet-4 |
| `ADVANCED` | Code generation, research, complex analysis | claude-sonnet-4 |
| `REASONING` | Deep reasoning, synthesis, complex problem solving | claude-sonnet-4 |

### Task-to-Tier Mapping

```python
TASK_TIER_MAPPING = {
    TaskType.CLASSIFICATION: ModelTier.BASIC,
    TaskType.SUMMARIZATION: ModelTier.BASIC,
    TaskType.SEARCH: ModelTier.STANDARD,
    TaskType.CODE_GENERATION: ModelTier.ADVANCED,
    TaskType.RESEARCH: ModelTier.ADVANCED,
    TaskType.REASONING: ModelTier.ADVANCED,
    TaskType.INNOVATION: ModelTier.REASONING,
}
```

### Configuration

Override default models with environment variables:

```bash
export LLM_TIER_BASIC_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0
export LLM_TIER_STANDARD_MODEL=us.anthropic.claude-sonnet-4-20250514-v1:0
export LLM_TIER_ADVANCED_MODEL=us.anthropic.claude-sonnet-4-20250514-v1:0
export LLM_TIER_REASONING_MODEL=us.anthropic.claude-sonnet-4-20250514-v1:0
```

### CLI Command

```bash
dova models  # Display current model tiering configuration
```

---

## 🌐 Web Search & Intelligent Source Selection

### Multi-Provider Web Search

DOVA supports multiple web search providers with automatic selection and fallback, enabling queries about current events, news, and real-time information without requiring any API keys.

**Providers (in priority order):**

| Provider | API Key Required | Features |
|----------|------------------|----------|
| **Brave Search** | Yes (`BRAVE_API_KEY`) | Structured results, freshness filtering |
| **Perplexity Sonar** | Yes (`PERPLEXITY_API_KEY`) | AI-synthesized answers with citations |
| **Tavily** | Yes (`TAVILY_API_KEY`) | Advanced search depth, relevance scoring |
| **DuckDuckGo** | No (free) | Always-available fallback, no setup needed |

**Auto-Selection Logic:**
- When `provider=auto` (default), DOVA tries providers in priority order
- Falls back to DuckDuckGo if no API keys are configured
- Web search works out of the box with zero configuration

**Configuration:**
```bash
# Optional - configure better providers for improved results
export BRAVE_API_KEY=xxx           # https://brave.com/search/api/
export PERPLEXITY_API_KEY=xxx      # https://perplexity.ai/settings/api
export TAVILY_API_KEY=xxx          # https://tavily.com

# Or use the MCP prefix
export MCP_TAVILY_API_KEY=tvly-xxxxxxxxxxxx
```

**Usage Example:**
```bash
# Works immediately with DuckDuckGo (no API key needed)
dova research "who did Trump nominate to replace Power?" -s web

# With Brave API key configured, uses Brave Search for better results
BRAVE_API_KEY=xxx dova research "latest AI news" -s web
```

### Intelligent Source Selection

The orchestrator automatically selects appropriate sources based on query analysis:

| Query Type | Indicators | Selected Sources |
|------------|------------|------------------|
| News/Current Events | "latest", "news", "nominated", "announced" | `web` |
| Technical/Research | "architecture", "implementation", "paper" | `arxiv`, `github` |
| ML Models | "model", "transformer", "training" | `huggingface`, `arxiv` |
| Mixed | Combination of indicators | Multiple sources |

**Implementation:**
```python
def _select_appropriate_sources(query: str, query_type: str) -> list[str]:
    """Select sources based on query analysis."""
    news_indicators = ["news", "latest", "announced", "nominated", ...]
    is_news_query = any(indicator in query.lower() for indicator in news_indicators)

    if is_news_query:
        return ["web"]  # Prioritize web for current events
    elif query_type == "research":
        return ["arxiv", "github", "huggingface"]
    else:
        return ["web", "arxiv", "github", "huggingface"]
```

### Enhanced Intent Classification

The orchestrator extracts key entities for better search queries:

```python
@dataclass
class ParsedIntent:
    intent: UserIntent
    confidence: float
    entities: dict[str, Any]  # topics, primary_subject, search_terms
    recommended_sources: list[str]  # Intelligent source recommendations
```

**Entity Fields:**
- `primary_subject`: The main subject of the query (e.g., model name, person)
- `search_terms`: Key terms extracted for search queries
- `topics`: Broader topics for context
- `recommended_sources`: Sources best suited for this query type

---

## 🏛️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              DOVA PLATFORM - HIGH LEVEL ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐         │
│  │   USER INTERFACE    │    │   API GATEWAY       │    │  AUTHENTICATION &   │         │
│  │   LAYER             │◄──►│   (Kong/Nginx)      │◄──►│  AUTHORIZATION      │         │
│  │                     │    │                     │    │  (OAuth 2.0/OIDC)   │         │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────────┘         │
│             │                          │                          │                     │
│             ▼                          ▼                          ▼                     │
│  ┌──────────────────────────────────────────────────────────────────────────────┐      │
│  │                         ORCHESTRATION LAYER                                   │      │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │      │
│  │  │  MASTER         │  │  TASK           │  │  WORKFLOW       │               │      │
│  │  │  ORCHESTRATOR   │◄►│  SCHEDULER      │◄►│  ENGINE         │               │      │
│  │  │  AGENT          │  │  (Temporal)     │  │  (Apache Airflow)│              │      │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘               │      │
│  └──────────────────────────────────────────────────────────────────────────────┘      │
│             │                          │                          │                     │
│             ▼                          ▼                          ▼                     │
│  ┌──────────────────────────────────────────────────────────────────────────────┐      │
│  │                    SPECIALIZED AGENT CLUSTERS                                 │      │
│  │                                                                               │      │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │      │
│  │  │ ACQUISITION  │ │ RESEARCH &   │ │ USER         │ │ SANDBOX &    │        │      │
│  │  │ AGENTS       │ │ INNOVATION   │ │ PROFILING    │ │ VALIDATION   │        │      │
│  │  │ CLUSTER      │ │ AGENTS       │ │ AGENTS       │ │ AGENTS       │        │      │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘        │      │
│  └──────────────────────────────────────────────────────────────────────────────┘      │
│             │                          │                          │                     │
│             ▼                          ▼                          ▼                     │
│  ┌──────────────────────────────────────────────────────────────────────────────┐      │
│  │                    MCP INTEGRATION & SOURCE REGISTRY LAYER                    │      │
│  │  ┌───────────────────────────────────────────────────────────────────────┐   │      │
│  │  │                     SOURCE REGISTRY (Quality Learning)                 │   │      │
│  │  └───────────────────────────────────────────────────────────────────────┘   │      │
│  │       │                        │                        │                     │      │
│  │  ┌────┴────────────────┐ ┌────┴────────────────┐ ┌─────┴───────────────┐    │      │
│  │  │   Built-in (MCP)    │ │   Custom Web/RSS    │ │   Custom APIs       │    │      │
│  │  └─────────────────────┘ └─────────────────────┘ └─────────────────────┘    │      │
│  │       │                                                                       │      │
│  │  ┌────┴────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐ ┌─────────────┐        │      │
│  │  │ ArXiv   │ │ GitHub  │ │ HF MCP  │ │ PubMed MCP  │ │ IEEE MCP    │        │      │
│  │  │ MCP     │ │ MCP     │ │         │ │             │ │             │        │      │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────────┘ └─────────────┘        │      │
│  └──────────────────────────────────────────────────────────────────────────────┘      │
│             │                          │                          │                     │
│             ▼                          ▼                          ▼                     │
│  ┌──────────────────────────────────────────────────────────────────────────────┐      │
│  │                         DATA & STORAGE LAYER                                  │      │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │      │
│  │  │ Vector DB   │ │ Graph DB    │ │ Time-Series │ │ Object      │            │      │
│  │  │ (Pinecone/  │ │ (Neo4j)     │ │ (InfluxDB)  │ │ Storage     │            │      │
│  │  │ Qdrant)     │ │             │ │             │ │ (MinIO/S3)  │            │      │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘            │      │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                            │      │
│  │  │ PostgreSQL  │ │ Redis       │ │ Elasticsearch│                           │      │
│  │  │ (Metadata)  │ │ (Cache)     │ │ (Search)     │                           │      │
│  │  └─────────────┘ └─────────────┘ └─────────────┘                            │      │
│  └──────────────────────────────────────────────────────────────────────────────┘      │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Detailed Component Design

### 1. **USER INTERFACE LAYER**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE COMPONENTS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────┐    ┌────────────────────────┐                   │
│  │   WEB APPLICATION      │    │   MOBILE APPS          │                   │
│  │   (React/Next.js)      │    │   (React Native)       │                   │
│  │   - Dashboard          │    │   - Push Notifications │                   │
│  │   - Research Explorer  │    │   - Quick Actions      │                   │
│  │   - Profile Settings   │    │   - Voice Interface    │                   │
│  │   - Collaboration Hub  │    │                        │                   │
│  └────────────────────────┘    └────────────────────────┘                   │
│                                                                              │
│  ┌────────────────────────┐    ┌────────────────────────┐                   │
│  │   IDE EXTENSIONS       │    │   API/SDK              │                   │
│  │   - VS Code Plugin     │    │   - REST API           │                   │
│  │   - JupyterLab Ext     │    │   - GraphQL API        │                   │
│  │   - CLI Tool           │    │   - Python/JS SDKs    │                   │
│  └────────────────────────┘    └────────────────────────┘                   │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │   CONVERSATIONAL INTERFACE                                      │         │
│  │   - Multi-turn Dialog Management                                │         │
│  │   - Context Preservation                                        │         │
│  │   - Intent Classification                                       │         │
│  │   - Slot Filling for Query Parameters                           │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation Technologies:**
- **Frontend**: Next.js 14+ with TypeScript, TailwindCSS, Shadcn/UI
- **Real-time**: WebSocket connections via Socket.io
- **State Management**: Zustand + React Query
- **Collaboration**: Y.js for real-time collaborative editing

---

### 2. **MASTER ORCHESTRATION LAYER**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MASTER ORCHESTRATOR AGENT (MOA)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    INTENT UNDERSTANDING MODULE                   │        │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │        │
│  │  │ Query        │ │ Context      │ │ User         │            │        │
│  │  │ Parser       │→│ Enricher     │→│ Intent       │            │        │
│  │  │ (NLU)        │ │              │ │ Classifier   │            │        │
│  │  └──────────────┘ └──────────────┘ └──────────────┘            │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                           │                                                  │
│                           ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    TASK DECOMPOSITION ENGINE                     │        │
│  │  • Hierarchical Task Network (HTN) Planning                      │        │
│  │  • Dependency Graph Construction                                 │        │
│  │  • Resource Estimation & Allocation                              │        │
│  │  • Priority Scoring (urgency × impact × user_preference)         │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                           │                                                  │
│                           ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    AGENT DISPATCH CONTROLLER                     │        │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │        │
│  │  │ Agent        │ │ Load         │ │ Result       │            │        │
│  │  │ Selection    │→│ Balancer     │→│ Aggregator   │            │        │
│  │  │ Algorithm    │ │              │ │              │            │        │
│  │  └──────────────┘ └──────────────┘ └──────────────┘            │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                           │                                                  │
│                           ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    COMMUNICATION BUS (A2A Protocol)              │        │
│  │  • Agent-to-Agent Protocol (A2A) Implementation                  │        │
│  │  • Message Queue (Apache Kafka / RabbitMQ)                       │        │
│  │  • Event-Driven Architecture                                     │        │
│  │  • Circuit Breaker Pattern for Fault Tolerance                   │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Orchestrator Logic (Pseudo-code):**

```python
class MasterOrchestratorAgent:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.task_decomposer = HTNPlanner()
        self.agent_registry = AgentRegistry()
        self.result_synthesizer = ResultSynthesizer()
    
    async def process_request(self, user_query: str, user_profile: UserProfile):
        # Step 1: Understand intent and context
        intent = await self.intent_classifier.classify(
            query=user_query,
            user_context=user_profile,
            conversation_history=self.get_history(user_profile.id)
        )
        
        # Step 2: Decompose into sub-tasks
        task_graph = await self.task_decomposer.plan(
            intent=intent,
            available_agents=self.agent_registry.get_available(),
            constraints=user_profile.preferences
        )
        
        # Step 3: Execute task graph (parallel where possible)
        results = await self.execute_task_graph(task_graph)
        
        # Step 4: Synthesize and personalize results
        final_output = await self.result_synthesizer.synthesize(
            results=results,
            user_profile=user_profile,
            output_format=intent.expected_output_type
        )
        
        return final_output
```

---

### 3. **SPECIALIZED AGENT CLUSTERS**

#### 3.1 **DATA ACQUISITION AGENT CLUSTER**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION AGENT CLUSTER                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    WEB CRAWLER AGENT                             │        │
│  │  • Distributed crawling with Scrapy Cluster                      │        │
│  │  • JavaScript rendering (Playwright/Puppeteer)                   │        │
│  │  • Rate limiting & robots.txt compliance                         │        │
│  │  • Content deduplication (SimHash)                               │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    ARXIV MONITOR AGENT                           │        │
│  │  • Real-time new paper detection (via ArXiv API/RSS)             │        │
│  │  • Category-specific subscriptions                               │        │
│  │  • Citation network analysis                                     │        │
│  │  • PDF parsing & extraction (GROBID + PyMuPDF)                   │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    GITHUB INTELLIGENCE AGENT                     │        │
│  │  • Repository monitoring (stars, forks, releases)                │        │
│  │  • Code change detection & analysis                              │        │
│  │  • Dependency vulnerability scanning                             │        │
│  │  • README/documentation extraction                               │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    HUGGINGFACE TRACKER AGENT                     │        │
│  │  • Model registry monitoring                                     │        │
│  │  • Dataset updates tracking                                      │        │
│  │  • Space discovery & evaluation                                  │        │
│  │  • Leaderboard changes detection                                 │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    ACADEMIC DATABASE AGENT                       │        │
│  │  • PubMed, IEEE Xplore, ACM DL connectors                        │        │
│  │  • Semantic Scholar integration                                  │        │
│  │  • Google Scholar monitoring (with rate limits)                  │        │
│  │  • Cross-reference resolution                                    │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    CONTENT PROCESSING PIPELINE                   │        │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │        │
│  │  │ Document │→│ Entity   │→│ Embedding│→│ Knowledge│           │        │
│  │  │ Parser   │ │ Extractor│ │ Generator│ │ Graph    │           │        │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 3.2 **RESEARCH & INNOVATION AGENT CLUSTER**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 RESEARCH & INNOVATION AGENT CLUSTER                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    LITERATURE SYNTHESIS AGENT                    │        │
│  │  • Systematic review automation                                  │        │
│  │  • Gap analysis in research landscape                            │        │
│  │  • Trend detection & forecasting                                 │        │
│  │  • Citation impact prediction                                    │        │
│  │  • Meta-analysis support                                         │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    HYPOTHESIS GENERATION AGENT                   │        │
│  │  • Cross-domain knowledge connection                             │        │
│  │  • Analogical reasoning engine                                   │        │
│  │  • Counterfactual scenario generation                            │        │
│  │  • Novel combination discovery                                   │        │
│  │  • Feasibility scoring                                           │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    SOLUTION ARCHITECT AGENT                      │        │
│  │  • Architecture pattern matching                                 │        │
│  │  • Component selection optimization                              │        │
│  │  • Trade-off analysis (cost, performance, scalability)           │        │
│  │  • Best practice recommendation                                  │        │
│  │  • Implementation roadmap generation                             │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    EXPERIMENTAL DESIGN AGENT                     │        │
│  │  • Experimental methodology suggestion                           │        │
│  │  • Statistical power analysis                                    │        │
│  │  • Baseline selection                                            │        │
│  │  • Metric recommendation                                         │        │
│  │  • Ablation study design                                         │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    DEBATE & CRITIQUE AGENTS (Bull vs Bear)       │        │
│  │  ┌─────────────────┐         ┌─────────────────┐                │        │
│  │  │ ADVOCATE AGENT  │◄───────►│ CRITIC AGENT    │                │        │
│  │  │ (Pro position)  │ Debate  │ (Counter args)  │                │        │
│  │  └─────────────────┘         └─────────────────┘                │        │
│  │              │                       │                           │        │
│  │              └───────────┬───────────┘                           │        │
│  │                          ▼                                       │        │
│  │              ┌─────────────────────┐                             │        │
│  │              │ SYNTHESIS AGENT     │                             │        │
│  │              │ (Balanced conclusion)│                            │        │
│  │              └─────────────────────┘                             │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 3.3 **USER PROFILING & PERSONALIZATION AGENT CLUSTER**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              USER PROFILING & PERSONALIZATION AGENT CLUSTER                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    USER PROFILE DATA MODEL                       │        │
│  │                                                                  │        │
│  │  {                                                               │        │
│  │    "user_id": "uuid",                                            │        │
│  │    "explicit_profile": {                                         │        │
│  │      "role": "ML Researcher",                                    │        │
│  │      "organization": "Enterprise Corp",                          │        │
│  │      "expertise_level": "senior",                                │        │
│  │      "declared_interests": ["NLP", "LLM", "Multi-Agent"]         │        │
│  │    },                                                            │        │
│  │    "implicit_profile": {                                         │        │
│  │      "interest_vector": [0.8, 0.6, ...],  // 768-dim embedding   │        │
│  │      "topic_affinities": {"transformers": 0.92, "RL": 0.67},     │        │
│  │      "reading_patterns": {                                       │        │
│  │        "preferred_depth": "technical",                           │        │
│  │        "avg_session_duration": 1200,                             │        │
│  │        "peak_activity_hours": [9, 10, 14, 15]                    │        │
│  │      },                                                          │        │
│  │      "interaction_history": [...],                               │        │
│  │      "feedback_signals": {...}                                   │        │
│  │    },                                                            │        │
│  │    "temporal_profile": {                                         │        │
│  │      "short_term_interests": [...],    // Last 7 days            │        │
│  │      "medium_term_interests": [...],   // Last 30 days           │        │
│  │      "long_term_interests": [...]      // Historical baseline    │        │
│  │    },                                                            │        │
│  │    "social_graph": {                                             │        │
│  │      "collaborators": [...],                                     │        │
│  │      "influence_network": [...]                                  │        │
│  │    }                                                             │        │
│  │  }                                                               │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    PROFILE LEARNING AGENT                        │        │
│  │  ┌──────────────────────────────────────────────────────────┐   │        │
│  │  │             INTERACTION CAPTURE MODULE                    │   │        │
│  │  │  • Query analysis & topic extraction                      │   │        │
│  │  │  • Click-through tracking                                 │   │        │
│  │  │  • Dwell time measurement                                 │   │        │
│  │  │  • Explicit feedback (likes, saves, shares)               │   │        │
│  │  │  • Implicit signals (scroll depth, copy actions)          │   │        │
│  │  └──────────────────────────────────────────────────────────┘   │        │
│  │                           │                                      │        │
│  │                           ▼                                      │        │
│  │  ┌──────────────────────────────────────────────────────────┐   │        │
│  │  │             PROFILE UPDATE ENGINE                         │   │        │
│  │  │  • Bayesian preference updating                           │   │        │
│  │  │  • Temporal decay functions                               │   │        │
│  │  │  • Interest drift detection                               │   │        │
│  │  │  • Contradiction resolution                               │   │        │
│  │  └──────────────────────────────────────────────────────────┘   │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    PERSONALIZATION ENGINE                        │        │
│  │  • Content ranking personalization                               │        │
│  │  • Query expansion based on profile                              │        │
│  │  • Explanation style adaptation                                  │        │
│  │  • Notification timing optimization                              │        │
│  │  • Serendipity injection (exploration vs exploitation)           │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    PROACTIVE RECOMMENDATION AGENT                │        │
│  │  ┌──────────────────────────────────────────────────────────┐   │        │
│  │  │             TRIGGER CONDITIONS                            │   │        │
│  │  │  • New paper in interest area                             │   │        │
│  │  │  • Trending topic alignment                               │   │        │
│  │  │  • Collaborator activity                                  │   │        │
│  │  │  • Research deadline approach                             │   │        │
│  │  │  • Gap in knowledge graph filled                          │   │        │
│  │  └──────────────────────────────────────────────────────────┘   │        │
│  │                           │                                      │        │
│  │                           ▼                                      │        │
│  │  ┌──────────────────────────────────────────────────────────┐   │        │
│  │  │             DELIVERY OPTIMIZATION                         │   │        │
│  │  │  • Channel selection (email, push, in-app)                │   │        │
│  │  │  • Timing optimization                                    │   │        │
│  │  │  • Frequency capping                                      │   │        │
│  │  │  • Urgency classification                                 │   │        │
│  │  └──────────────────────────────────────────────────────────┘   │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 3.4 **SANDBOX & VALIDATION AGENT CLUSTER**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SANDBOX & VALIDATION AGENT CLUSTER                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    SANDBOX ENVIRONMENT MANAGER                   │        │
│  │                                                                  │        │
│  │  ┌────────────────────────────────────────────────────────────┐ │        │
│  │  │              CONTAINER ORCHESTRATION (Kubernetes)          │ │        │
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │ │        │
│  │  │  │ Python   │ │ Node.js  │ │ JVM      │ │ Custom   │     │ │        │
│  │  │  │ Sandbox  │ │ Sandbox  │ │ Sandbox  │ │ ML Envs  │     │ │        │
│  │  │  │ (3.9-12) │ │ (18-22)  │ │ (11-21)  │ │ (PyTorch │     │ │        │
│  │  │  │          │ │          │ │          │ │  TF, JAX)│     │ │        │
│  │  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘     │ │        │
│  │  └────────────────────────────────────────────────────────────┘ │        │
│  │                                                                  │        │
│  │  • Resource quotas (CPU, Memory, GPU, Time)                     │        │
│  │  • Network isolation (egress whitelisting)                      │        │
│  │  • File system sandboxing (ephemeral volumes)                   │        │
│  │  • Secrets management (Vault integration)                       │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    CODE GENERATION & VALIDATION AGENT            │        │
│  │                                                                  │        │
│  │  ┌────────────────────────────────────────────────────────────┐ │        │
│  │  │              CODE GENERATION PIPELINE                       │ │        │
│  │  │                                                             │ │        │
│  │  │  Specification → Code Gen → Static Analysis → Test Gen     │ │        │
│  │  │       │             │             │              │          │ │        │
│  │  │       ▼             ▼             ▼              ▼          │ │        │
│  │  │  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐        │ │        │
│  │  │  │ Intent │   │ Multi- │   │ Linters│   │ Pytest │        │ │        │
│  │  │  │ Parser │   │ LLM    │   │ Type   │   │ Unit   │        │ │        │
│  │  │  │        │   │ Code   │   │ Check  │   │ Tests  │        │ │        │
│  │  │  │        │   │ Gen    │   │        │   │        │        │ │        │
│  │  │  └────────┘   └────────┘   └────────┘   └────────┘        │ │        │
│  │  └────────────────────────────────────────────────────────────┘ │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    EXPERIMENT EXECUTION AGENT                    │        │
│  │                                                                  │        │
│  │  Workflow:                                                       │        │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │        │
│  │  │ Setup   │→ │ Execute │→ │ Monitor │→ │ Cleanup │            │        │
│  │  │ Env     │  │ Code    │  │ & Log   │  │ & Report│            │        │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘            │        │
│  │                                                                  │        │
│  │  Features:                                                       │        │
│  │  • Reproducibility tracking (MLflow/W&B integration)             │        │
│  │  • Checkpoint & resume capability                                │        │
│  │  • Distributed execution support                                 │        │
│  │  • Cost estimation & budgeting                                   │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    SOLUTION VALIDATION AGENT                     │        │
│  │                                                                  │        │
│  │  Validation Dimensions:                                          │        │
│  │  ┌─────────────────────────────────────────────────────────┐    │        │
│  │  │ FUNCTIONAL    │ Does it produce correct outputs?        │    │        │
│  │  ├───────────────┼─────────────────────────────────────────┤    │        │
│  │  │ PERFORMANCE   │ Latency, throughput, resource usage     │    │        │
│  │  ├───────────────┼─────────────────────────────────────────┤    │        │
│  │  │ SCALABILITY   │ Behavior under increased load           │    │        │
│  │  ├───────────────┼─────────────────────────────────────────┤    │        │
│  │  │ SECURITY      │ Vulnerability scanning, input sanitization│   │        │
│  │  ├───────────────┼─────────────────────────────────────────┤    │        │
│  │  │ RELIABILITY   │ Error handling, edge cases              │    │        │
│  │  ├───────────────┼─────────────────────────────────────────┤    │        │
│  │  │ MAINTAINABILITY│ Code quality, documentation            │    │        │
│  │  └─────────────────────────────────────────────────────────┘    │        │
│  │                                                                  │        │
│  │  Output: Confidence Score + Detailed Report + Recommendations    │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    DRY-RUN SIMULATION AGENT                      │        │
│  │                                                                  │        │
│  │  • Architecture simulation (component interaction modeling)      │        │
│  │  • Load simulation (synthetic traffic generation)                │        │
│  │  • Failure mode injection (chaos engineering)                    │        │
│  │  • Cost projection (cloud resource estimation)                   │        │
│  │  • Timeline estimation (effort prediction)                       │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 4. **MCP INTEGRATION LAYER**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MCP INTEGRATION LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    MCP GATEWAY (IBM Context Forge Pattern)       │        │
│  │                                                                  │        │
│  │  Features:                                                       │        │
│  │  • Unified API for all MCP servers                               │        │
│  │  • Protocol translation (stdio ↔ SSE ↔ HTTP)                     │        │
│  │  • Authentication & authorization layer                          │        │
│  │  • Rate limiting & quota management                              │        │
│  │  • Request caching & deduplication                               │        │
│  │  • Observability (metrics, tracing, logging)                     │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    MCP SERVER REGISTRY                           │        │
│  │                                                                  │        │
│  │  ┌────────────────────────────────────────────────────────────┐ │        │
│  │  │ Server          │ Capabilities        │ Status │ Priority │ │        │
│  │  ├─────────────────┼─────────────────────┼────────┼──────────┤ │        │
│  │  │ Web Search MCP  │ search, scrape      │ Active │ High     │ │        │
│  │  │ ArXiv MCP       │ search, fetch_paper │ Active │ High     │ │        │
│  │  │ GitHub MCP      │ repo_*, issue_*,    │ Active │ High     │ │        │
│  │  │                 │ pr_*, code_search   │        │          │ │        │
│  │  │ HuggingFace MCP │ model_*, dataset_*, │ Active │ High     │ │        │
│  │  │                 │ paper_*, space_*    │        │          │ │        │
│  │  │ PubMed MCP      │ search, fetch       │ Active │ Medium   │ │        │
│  │  │ IEEE MCP        │ search, fetch       │ Active │ Medium   │ │        │
│  │  │ Semantic Scholar│ search, citations   │ Active │ Medium   │ │        │
│  │  │ Notion MCP      │ read, write         │ Active │ Low      │ │        │
│  │  │ Slack MCP       │ notify, search      │ Active │ Low      │ │        │
│  │  └────────────────────────────────────────────────────────────┘ │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    USER SOURCE REGISTRY                          │        │
│  │                                                                  │        │
│  │  Beyond built-in MCP servers, users can add custom sources:      │        │
│  │                                                                  │        │
│  │  Source Types:                                                   │        │
│  │  • web_url  - Scrape web pages (blogs, docs, news sites)        │        │
│  │  • rss_feed - Parse RSS/Atom feeds (publication updates)        │        │
│  │  • api      - Call custom HTTP APIs (internal services)         │        │
│  │                                                                  │        │
│  │  Quality Learning:                                               │        │
│  │  • Implicit signals: queries, clicks, saves, result position    │        │
│  │  • Quality score: 0.0-1.0 (influences result ranking)           │        │
│  │  • Per-user storage: sources are private to each user           │        │
│  │                                                                  │        │
│  │  ┌────────────────────────────────────────────────────────────┐ │        │
│  │  │ Type     │ Example URL                 │ Auth Support      │ │        │
│  │  ├──────────┼─────────────────────────────┼───────────────────┤ │        │
│  │  │ web_url  │ https://blog.example.com    │ Headers           │ │        │
│  │  │ rss_feed │ https://news.ycombinator... │ None              │ │        │
│  │  │ api      │ https://api.corp.com/{query}│ Bearer, API Key   │ │        │
│  │  └────────────────────────────────────────────────────────────┘ │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    MCP TOOL INVOCATION PATTERN                   │        │
│  │                                                                  │        │
│  │  async def invoke_mcp_tool(server: str, tool: str, params: dict):│        │
│  │      # 1. Resolve server endpoint                                │        │
│  │      endpoint = mcp_registry.get_endpoint(server)                │        │
│  │                                                                  │        │
│  │      # 2. Check capability & permissions                         │        │
│  │      if not await check_permission(server, tool, current_user):  │        │
│  │          raise PermissionDenied()                                │        │
│  │                                                                  │        │
│  │      # 3. Apply rate limiting                                    │        │
│  │      await rate_limiter.acquire(server, tool)                    │        │
│  │                                                                  │        │
│  │      # 4. Execute with retry & circuit breaker                   │        │
│  │      async with circuit_breaker(server):                         │        │
│  │          result = await mcp_client.call(endpoint, tool, params)  │        │
│  │                                                                  │        │
│  │      # 5. Transform & cache result                               │        │
│  │      processed = transform_result(result, tool)                  │        │
│  │      await cache.set(cache_key(server, tool, params), processed) │        │
│  │                                                                  │        │
│  │      return processed                                            │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 5. **DATA & STORAGE LAYER**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA & STORAGE LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    VECTOR DATABASE (Qdrant/Pinecone)             │        │
│  │                                                                  │        │
│  │  Collections:                                                    │        │
│  │  • papers_embeddings (768D, ~10M vectors)                        │        │
│  │  • code_embeddings (768D, ~50M vectors)                          │        │
│  │  • user_interest_vectors (768D, ~100K vectors)                   │        │
│  │  • query_embeddings (768D, ~1M vectors)                          │        │
│  │                                                                  │        │
│  │  Features:                                                       │        │
│  │  • HNSW indexing for fast ANN search                             │        │
│  │  • Payload filtering (metadata + vector search)                  │        │
│  │  • Multi-tenancy support                                         │        │
│  │  • Quantization for memory efficiency                            │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    KNOWLEDGE GRAPH (Neo4j)                       │        │
│  │                                                                  │        │
│  │  Node Types:                                                     │        │
│  │  • Paper, Author, Institution, Topic, Method, Dataset            │        │
│  │  • Repository, Model, User, Query                                │        │
│  │                                                                  │        │
│  │  Relationship Types:                                             │        │
│  │  • CITES, AUTHORED_BY, AFFILIATED_WITH                           │        │
│  │  • USES_METHOD, TRAINED_ON, IMPLEMENTS                           │        │
│  │  • INTERESTED_IN, QUERIED, SAVED                                 │        │
│  │                                                                  │        │
│  │  ┌────────────────────────────────────────────────────────────┐ │        │
│  │  │         EXAMPLE KNOWLEDGE GRAPH SCHEMA                     │ │        │
│  │  │                                                             │ │        │
│  │  │     (User)──[INTERESTED_IN]──►(Topic)                      │ │        │
│  │  │        │                         │                          │ │        │
│  │  │   [QUERIED]              [RELATED_TO]                       │ │        │
│  │  │        │                         │                          │ │        │
│  │  │        ▼                         ▼                          │ │        │
│  │  │    (Paper)◄──[CITES]──(Paper)──[USES]──►(Method)           │ │        │
│  │  │        │                                    │               │ │        │
│  │  │   [AUTHORED_BY]                    [IMPLEMENTED_IN]         │ │        │
│  │  │        │                                    │               │ │        │
│  │  │        ▼                                    ▼               │ │        │
│  │  │   (Author)                           (Repository)           │ │        │
│  │  │                                            │                │ │        │
│  │  │                                      [HAS_MODEL]            │ │        │
│  │  │                                            │                │ │        │
│  │  │                                            ▼                │ │        │
│  │  │                                        (Model)              │ │        │
│  │  └────────────────────────────────────────────────────────────┘ │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    POSTGRESQL (Metadata & Transactions)          │        │
│  │                                                                  │        │
│  │  Tables:                                                         │        │
│  │  • users, user_profiles, user_sessions                           │        │
│  │  • queries, query_results, feedback                              │        │
│  │  • agents, agent_tasks, agent_logs                               │        │
│  │  • papers_metadata, repos_metadata                               │        │
│  │  • recommendations, notifications                                │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    REDIS (Caching & Real-time)                   │        │
│  │                                                                  │        │
│  │  Use Cases:                                                      │        │
│  │  • Session management                                            │        │
│  │  • Query result caching (TTL: 1hr)                               │        │
│  │  • Rate limiting counters                                        │        │
│  │  • Real-time agent status                                        │        │
│  │  • Pub/Sub for notifications                                     │        │
│  │  • Distributed locks                                             │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    ELASTICSEARCH (Full-text Search)              │        │
│  │                                                                  │        │
│  │  Indices:                                                        │        │
│  │  • papers (title, abstract, full_text, authors, keywords)        │        │
│  │  • code (content, documentation, comments)                       │        │
│  │  • conversations (query, response, timestamps)                   │        │
│  │                                                                  │        │
│  │  Features:                                                       │        │
│  │  • BM25 + semantic hybrid search                                 │        │
│  │  • Faceted navigation                                            │        │
│  │  • Highlighting & snippets                                       │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    OBJECT STORAGE (MinIO/S3)                     │        │
│  │                                                                  │        │
│  │  Buckets:                                                        │        │
│  │  • raw-papers (PDFs, HTML snapshots)                             │        │
│  │  • processed-papers (parsed JSON, extracted figures)             │        │
│  │  • code-artifacts (repositories, notebooks)                      │        │
│  │  • user-uploads                                                  │        │
│  │  • sandbox-artifacts (experiment outputs)                        │        │
│  │  • model-checkpoints                                             │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow Diagrams

### 5.1 **Query-Driven Research Flow**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     QUERY-DRIVEN RESEARCH DATA FLOW                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐                                                                │
│  │   User   │                                                                │
│  └────┬─────┘                                                                │
│       │ 1. Submit Query                                                      │
│       │    "Find latest advances in multi-agent LLM systems                  │
│       │     with code implementation for enterprise deployment"              │
│       ▼                                                                      │
│  ┌──────────────────────┐                                                    │
│  │   API Gateway        │                                                    │
│  │   • Auth validation  │                                                    │
│  │   • Rate limiting    │                                                    │
│  └────────┬─────────────┘                                                    │
│           │ 2. Authenticated Request                                         │
│           ▼                                                                  │
│  ┌──────────────────────┐      ┌──────────────────────┐                     │
│  │   Master Orchestrator│◄────►│   User Profile Store │                     │
│  │   Agent              │      │   (Redis + PostgreSQL)│                    │
│  │   • Parse intent     │      └──────────────────────┘                     │
│  │   • Load user context│              │                                     │
│  │   • Plan task graph  │              │ 3. Retrieve profile                 │
│  └────────┬─────────────┘              │    & preferences                    │
│           │                            │                                     │
│           │ 4. Decomposed Tasks        │                                     │
│           ▼                            │                                     │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │                    PARALLEL EXECUTION PHASE                     │         │
│  │                                                                 │         │
│  │  Task 1: ArXiv Search    Task 2: GitHub Search   Task 3: HF    │         │
│  │  ┌─────────────────┐     ┌─────────────────┐    ┌───────────┐  │         │
│  │  │ ArXiv MCP       │     │ GitHub MCP      │    │ HF MCP    │  │         │
│  │  │ • search_arxiv  │     │ • search_repos  │    │ • paper_  │  │         │
│  │  │ • search_by_    │     │ • search_code   │    │   search  │  │         │
│  │  │   category      │     │ • get_file_     │    │ • model_  │  │         │
│  │  │                 │     │   contents      │    │   search  │  │         │
│  │  └────────┬────────┘     └────────┬────────┘    └─────┬─────┘  │         │
│  │           │                       │                   │        │         │
│  └───────────┼───────────────────────┼───────────────────┼────────┘         │
│              │                       │                   │                   │
│              │ 5. Raw Results        │                   │                   │
│              ▼                       ▼                   ▼                   │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                    RESULT AGGREGATION                         │           │
│  │                                                               │           │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐    │           │
│  │  │ Deduplication │→ │ Relevance     │→ │ Personalized  │    │           │
│  │  │ & Merging     │  │ Ranking       │  │ Filtering     │    │           │
│  │  └───────────────┘  └───────────────┘  └───────────────┘    │           │
│  │           │                                                   │           │
│  │           ▼                                                   │           │
│  │  ┌───────────────────────────────────────────────────────┐   │           │
│  │  │              SYNTHESIS & ENRICHMENT                    │   │           │
│  │  │  • Cross-reference papers ↔ code ↔ models              │   │           │
│  │  │  • Extract key findings                                │   │           │
│  │  │  • Generate summary (LLM-powered)                      │   │           │
│  │  │  • Add actionable insights                             │   │           │
│  │  └───────────────────────────────────────────────────────┘   │           │
│  └───────────────────────────────────────────────────────────────┘           │
│                         │                                                    │
│                         │ 6. Synthesized Response                            │
│                         ▼                                                    │
│  ┌──────────────────────────────────────────────────────────────┐           │
│  │                    RESPONSE PERSONALIZATION                   │           │
│  │  • Adapt explanation depth to user expertise                  │           │
│  │  • Highlight items matching user interests                    │           │
│  │  • Format according to user preferences                       │           │
│  └──────────────────────────────────────────────────────────────┘           │
│                         │                                                    │
│                         │ 7. Final Response                                  │
│                         ▼                                                    │
│  ┌──────────┐                                                                │
│  │   User   │  + Update interaction history in profile                       │
│  └──────────┘                                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 **Proactive Recommendation Flow**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROACTIVE RECOMMENDATION DATA FLOW                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    CONTINUOUS MONITORING LAYER                   │        │
│  │                                                                  │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │        │
│  │  │ ArXiv RSS    │  │ GitHub       │  │ HF Model     │           │        │
│  │  │ Subscriber   │  │ Webhooks     │  │ Registry     │           │        │
│  │  │ (Every 1hr)  │  │ (Real-time)  │  │ (Every 6hr)  │           │        │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │        │
│  │         │                 │                 │                    │        │
│  │         └────────────────┬┴─────────────────┘                    │        │
│  │                          ▼                                       │        │
│  │              ┌───────────────────────┐                           │        │
│  │              │   New Content Queue   │                           │        │
│  │              │   (Kafka Topic)       │                           │        │
│  │              └───────────┬───────────┘                           │        │
│  └──────────────────────────┼───────────────────────────────────────┘        │
│                             │                                                │
│                             ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    CONTENT PROCESSING PIPELINE                   │        │
│  │                                                                  │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │        │
│  │  │ Parse &      │→ │ Generate     │→ │ Extract      │           │        │
│  │  │ Normalize    │  │ Embeddings   │  │ Entities     │           │        │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │        │
│  │                                              │                   │        │
│  │                                              ▼                   │        │
│  │                              ┌──────────────────────────┐        │        │
│  │                              │ Update Knowledge Graph   │        │        │
│  │                              │ & Vector Store           │        │        │
│  │                              └──────────────────────────┘        │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                             │                                                │
│                             ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    USER MATCHING ENGINE                          │        │
│  │                                                                  │        │
│  │  For each new_item:                                              │        │
│  │  ┌────────────────────────────────────────────────────────────┐ │        │
│  │  │ 1. Vector similarity search against user_interest_vectors  │ │        │
│  │  │    → Candidates with similarity > 0.75                     │ │        │
│  │  │                                                             │ │        │
│  │  │ 2. Graph-based relevance (paths in knowledge graph)        │ │        │
│  │  │    → Users connected via topics, authors, methods          │ │        │
│  │  │                                                             │ │        │
│  │  │ 3. Explicit subscription matching                          │ │        │
│  │  │    → Users following specific topics/authors/repos         │ │        │
│  │  │                                                             │ │        │
│  │  │ 4. Collaborative filtering                                 │ │        │
│  │  │    → Similar users liked similar content                   │ │        │
│  │  └────────────────────────────────────────────────────────────┘ │        │
│  │                             │                                    │        │
│  │                             ▼                                    │        │
│  │  ┌────────────────────────────────────────────────────────────┐ │        │
│  │  │              RELEVANCE SCORING                              │ │        │
│  │  │                                                             │ │        │
│  │  │  score = w1 * semantic_sim +                                │ │        │
│  │  │          w2 * graph_relevance +                             │ │        │
│  │  │          w3 * recency_boost +                               │ │        │
│  │  │          w4 * source_authority +                            │ │        │
│  │  │          w5 * novelty_factor                                │ │        │
│  │  │                                                             │ │        │
│  │  │  Filter: score > user_threshold (default: 0.7)              │ │        │
│  │  └────────────────────────────────────────────────────────────┘ │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                             │                                                │
│                             ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    DELIVERY OPTIMIZATION                         │        │
│  │                                                                  │        │
│  │  ┌────────────────────────────────────────────────────────────┐ │        │
│  │  │ Batch recommendations by user                               │ │        │
│  │  │                         │                                   │ │        │
│  │  │                         ▼                                   │ │        │
│  │  │ ┌─────────────────────────────────────────────────────┐    │ │        │
│  │  │ │ For each user_batch:                                │    │ │        │
│  │  │ │ • Check notification preferences                    │    │ │        │
│  │  │ │ • Calculate optimal delivery time                   │    │ │        │
│  │  │ │ • Apply frequency capping                           │    │ │        │
│  │  │ │ • Select channel (email/push/in-app)                │    │ │        │
│  │  │ │ • Generate personalized summary                     │    │ │        │
│  │  │ └─────────────────────────────────────────────────────┘    │ │        │
│  │  └────────────────────────────────────────────────────────────┘ │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                             │                                                │
│                             ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    DELIVERY CHANNELS                             │        │
│  │                                                                  │        │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐│        │
│  │  │ Email      │  │ Push       │  │ In-App     │  │ Slack/     ││        │
│  │  │ (Daily     │  │ Notification│  │ Feed       │  │ Teams      ││        │
│  │  │  Digest)   │  │ (Urgent)   │  │ (Real-time)│  │ Integration││        │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘│        │
│  │                                                                  │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 **Innovation & Validation Flow**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INNOVATION & VALIDATION DATA FLOW                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    PHASE 1: PROBLEM ANALYSIS                      │       │
│  │                                                                   │       │
│  │  User Request: "Design an efficient RAG system for                │       │
│  │                 multi-modal enterprise documents"                 │       │
│  │                                                                   │       │
│  │  ┌─────────────────┐   ┌─────────────────┐                       │       │
│  │  │ Problem         │   │ Constraint      │                       │       │
│  │  │ Decomposition   │──►│ Extraction      │                       │       │
│  │  │ Agent           │   │ Agent           │                       │       │
│  │  └─────────────────┘   └────────┬────────┘                       │       │
│  │                                 │                                 │       │
│  │                                 ▼                                 │       │
│  │                    ┌─────────────────────────┐                    │       │
│  │                    │ Problem Specification   │                    │       │
│  │                    │ • Goals & objectives    │                    │       │
│  │                    │ • Constraints           │                    │       │
│  │                    │ • Success metrics       │                    │       │
│  │                    │ • Scope boundaries      │                    │       │
│  │                    └─────────────────────────┘                    │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                 │                                            │
│                                 ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    PHASE 2: LITERATURE SYNTHESIS                  │       │
│  │                                                                   │       │
│  │  ┌─────────────────────────────────────────────────────────┐     │       │
│  │  │              PARALLEL RESEARCH QUERIES                   │     │       │
│  │  │                                                          │     │       │
│  │  │  ArXiv: "RAG retrieval augmented generation multimodal"  │     │       │
│  │  │  GitHub: "multimodal RAG enterprise implementation"      │     │       │
│  │  │  HuggingFace: "multimodal embedding models"              │     │       │
│  │  │  Semantic Scholar: "document understanding retrieval"    │     │       │
│  │  └─────────────────────────────────────────────────────────┘     │       │
│  │                                 │                                 │       │
│  │                                 ▼                                 │       │
│  │  ┌─────────────────────────────────────────────────────────┐     │       │
│  │  │              LITERATURE SYNTHESIS AGENT                  │     │       │
│  │  │                                                          │     │       │
│  │  │  • Identify SOTA approaches                              │     │       │
│  │  │  • Map technique landscape                               │     │       │
│  │  │  • Extract key innovations                               │     │       │
│  │  │  • Note limitations & gaps                               │     │       │
│  │  │  • Compile reference implementations                     │     │       │
│  │  └─────────────────────────────────────────────────────────┘     │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                 │                                            │
│                                 ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    PHASE 3: SOLUTION GENERATION                   │       │
│  │                                                                   │       │
│  │  ┌─────────────────────────────────────────────────────────┐     │       │
│  │  │              HYPOTHESIS GENERATION AGENT                 │     │       │
│  │  │                                                          │     │       │
│  │  │  Generates multiple solution candidates:                 │     │       │
│  │  │  • Candidate A: ColPali-based late interaction          │     │       │
│  │  │  • Candidate B: Hybrid chunking + vision encoder        │     │       │
│  │  │  • Candidate C: Graph-based document structure RAG      │     │       │
│  │  └─────────────────────────────────────────────────────────┘     │       │
│  │                                 │                                 │       │
│  │                                 ▼                                 │       │
│  │  ┌─────────────────────────────────────────────────────────┐     │       │
│  │  │              DEBATE & CRITIQUE MECHANISM                 │     │       │
│  │  │                                                          │     │       │
│  │  │  ┌─────────────┐         ┌─────────────┐                │     │       │
│  │  │  │ ADVOCATE    │◄───────►│ CRITIC      │                │     │       │
│  │  │  │ AGENT       │ Round 1 │ AGENT       │                │     │       │
│  │  │  │             │ Round 2 │             │                │     │       │
│  │  │  │ "ColPali is │ Round 3 │ "But high   │                │     │       │
│  │  │  │  simpler..."│         │  latency..."│                │     │       │
│  │  │  └─────────────┘         └─────────────┘                │     │       │
│  │  │              │                   │                       │     │       │
│  │  │              └─────────┬─────────┘                       │     │       │
│  │  │                        ▼                                 │     │       │
│  │  │              ┌─────────────────┐                         │     │       │
│  │  │              │ SYNTHESIS AGENT │                         │     │       │
│  │  │              │ Balanced final  │                         │     │       │
│  │  │              │ recommendation  │                         │     │       │
│  │  │              └─────────────────┘                         │     │       │
│  │  └─────────────────────────────────────────────────────────┘     │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                 │                                            │
│                                 ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    PHASE 4: SOLUTION ARCHITECTURE                 │       │
│  │                                                                   │       │
│  │  ┌─────────────────────────────────────────────────────────┐     │       │
│  │  │              SOLUTION ARCHITECT AGENT                    │     │       │
│  │  │                                                          │     │       │
│  │  │  Produces:                                               │     │       │
│  │  │  • System architecture diagram                           │     │       │
│  │  │  • Component specifications                              │     │       │
│  │  │  • Data flow diagrams                                    │     │       │
│  │  │  • Technology stack recommendations                      │     │       │
│  │  │  • Implementation phases                                 │     │       │
│  │  │  • Resource estimates                                    │     │       │
│  │  └─────────────────────────────────────────────────────────┘     │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                 │                                            │
│                                 ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    PHASE 5: SANDBOX VALIDATION                    │       │
│  │                                                                   │       │
│  │  ┌─────────────────────────────────────────────────────────┐     │       │
│  │  │              CODE GENERATION AGENT                       │     │       │
│  │  │                                                          │     │       │
│  │  │  Generates:                                              │     │       │
│  │  │  • Proof-of-concept implementation                       │     │       │
│  │  │  • Unit tests                                            │     │       │
│  │  │  • Benchmark scripts                                     │     │       │
│  │  └─────────────────────────────────────────────────────────┘     │       │
│  │                                 │                                 │       │
│  │                                 ▼                                 │       │
│  │  ┌─────────────────────────────────────────────────────────┐     │       │
│  │  │              SANDBOX EXECUTION                           │     │       │
│  │  │                                                          │     │       │
│  │  │  ┌──────────────────────────────────────────────────┐   │     │       │
│  │  │  │ Kubernetes Sandbox Pod                           │   │     │       │
│  │  │  │ • CPU: 8 cores, RAM: 32GB, GPU: 1x A100         │   │     │       │
│  │  │  │ • Timeout: 30 minutes                            │   │     │       │
│  │  │  │ • Network: egress restricted                     │   │     │       │
│  │  │  │                                                  │   │     │       │
│  │  │  │  Run: POC code + unit tests + benchmarks         │   │     │       │
│  │  │  └──────────────────────────────────────────────────┘   │     │       │
│  │  └─────────────────────────────────────────────────────────┘     │       │
│  │                                 │                                 │       │
│  │                                 ▼                                 │       │
│  │  ┌─────────────────────────────────────────────────────────┐     │       │
│  │  │              VALIDATION REPORT                           │     │       │
│  │  │                                                          │     │       │
│  │  │  • Functional: ✅ 95% tests passing                      │     │       │
│  │  │  • Performance: Latency 120ms (target: <200ms) ✅        │     │       │
│  │  │  • Scalability: Linear up to 1000 QPS ✅                 │     │       │
│  │  │  • Resource: 12GB RAM peak (within budget) ✅            │     │       │
│  │  │  • Security: No vulnerabilities detected ✅              │     │       │
│  │  │                                                          │     │       │
│  │  │  Overall Confidence: 87% (High)                          │     │       │
│  │  └─────────────────────────────────────────────────────────┘     │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                 │                                            │
│                                 ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    PHASE 6: FINAL RECOMMENDATION                  │       │
│  │                                                                   │       │
│  │  ┌─────────────────────────────────────────────────────────────┐ │       │
│  │  │                    DELIVERABLES                              │ │       │
│  │  │                                                              │ │       │
│  │  │  📄 Technical Report (PDF)                                   │ │       │
│  │  │     • Executive summary                                      │ │       │
│  │  │     • Problem analysis                                       │ │       │
│  │  │     • Solution design                                        │ │       │
│  │  │     • Validation results                                     │ │       │
│  │  │     • Implementation roadmap                                 │ │       │
│  │  │                                                              │ │       │
│  │  │  📦 Code Artifacts                                           │ │       │
│  │  │     • GitHub repository with POC                             │ │       │
│  │  │     • Documentation                                          │ │       │
│  │  │     • Deployment configs                                     │ │       │
│  │  │                                                              │ │       │
│  │  │  📊 Presentation Deck                                        │ │       │
│  │  │     • For stakeholder communication                          │ │       │
│  │  └─────────────────────────────────────────────────────────────┘ │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🏢 Enterprise Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE DEPLOYMENT ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    KUBERNETES CLUSTER (Multi-AZ)                 │        │
│  │                                                                  │        │
│  │  ┌─────────────────────────────────────────────────────────┐    │        │
│  │  │              CONTROL PLANE NAMESPACE                     │    │        │
│  │  │  • Master Orchestrator (3 replicas, HA)                  │    │        │
│  │  │  • API Gateway (Kong, 5 replicas)                        │    │        │
│  │  │  • Workflow Engine (Temporal, 3 workers)                 │    │        │
│  │  │  • MCP Gateway (3 replicas)                              │    │        │
│  │  └─────────────────────────────────────────────────────────┘    │        │
│  │                                                                  │        │
│  │  ┌─────────────────────────────────────────────────────────┐    │        │
│  │  │              AGENT NAMESPACE                             │    │        │
│  │  │  • Acquisition Agents (HPA: 3-20 pods)                   │    │        │
│  │  │  • Research Agents (HPA: 2-10 pods)                      │    │        │
│  │  │  • Profiling Agents (HPA: 2-8 pods)                      │    │        │
│  │  │  • Sandbox Agents (HPA: 1-5 pods, GPU-enabled)           │    │        │
│  │  └─────────────────────────────────────────────────────────┘    │        │
│  │                                                                  │        │
│  │  ┌─────────────────────────────────────────────────────────┐    │        │
│  │  │              DATA NAMESPACE                              │    │        │
│  │  │  • PostgreSQL (Primary + 2 Replicas, PGBouncer)          │    │        │
│  │  │  • Redis Cluster (6 nodes)                               │    │        │
│  │  │  • Elasticsearch (3 masters, 5 data, 2 ingest)           │    │        │
│  │  └─────────────────────────────────────────────────────────┘    │        │
│  │                                                                  │        │
│  │  ┌─────────────────────────────────────────────────────────┐    │        │
│  │  │              SANDBOX NAMESPACE (Isolated)                │    │        │
│  │  │  • Ephemeral pods with strict resource quotas            │    │        │
│  │  │  • Network policies (egress whitelist only)              │    │        │
│  │  │  • GPU node pool (A100/H100 spot instances)              │    │        │
│  │  └─────────────────────────────────────────────────────────┘    │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    MANAGED SERVICES                              │        │
│  │                                                                  │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │        │
│  │  │ Vector DB    │  │ Neo4j Aura   │  │ S3/MinIO     │           │        │
│  │  │ (Pinecone/   │  │ (Graph DB)   │  │ (Object      │           │        │
│  │  │  Qdrant      │  │              │  │  Storage)    │           │        │
│  │  │  Cloud)      │  │              │  │              │           │        │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │        │
│  │                                                                  │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │        │
│  │  │ Kafka        │  │ Vault        │  │ LLM APIs     │           │        │
│  │  │ (MSK/        │  │ (Secrets)    │  │ (OpenAI,     │           │        │
│  │  │  Confluent)  │  │              │  │  Anthropic,  │           │        │
│  │  │              │  │              │  │  HF Inference)│          │        │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    OBSERVABILITY STACK                           │        │
│  │                                                                  │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │        │
│  │  │ Prometheus   │  │ Grafana      │  │ Jaeger       │           │        │
│  │  │ (Metrics)    │  │ (Dashboards) │  │ (Tracing)    │           │        │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │        │
│  │                                                                  │        │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │        │
│  │  │ ELK Stack    │  │ PagerDuty    │  │ OpenTelemetry│           │        │
│  │  │ (Logging)    │  │ (Alerting)   │  │ (Tracing)    │           │        │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔐 Security Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SECURITY ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    IDENTITY & ACCESS MANAGEMENT                  │        │
│  │                                                                  │        │
│  │  • SSO Integration (SAML 2.0 / OIDC)                             │        │
│  │  • Role-Based Access Control (RBAC)                              │        │
│  │    - Admin: Full platform access                                 │        │
│  │    - Researcher: Query + Innovation features                     │        │
│  │    - Viewer: Read-only access                                    │        │
│  │  • Attribute-Based Access Control (ABAC)                         │        │
│  │    - Department-specific data access                             │        │
│  │    - Project-level permissions                                   │        │
│  │  • Multi-Factor Authentication (MFA)                             │        │
│  │  • API Key Management with scopes                                │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    DATA SECURITY                                 │        │
│  │                                                                  │        │
│  │  • Encryption at Rest (AES-256)                                  │        │
│  │  • Encryption in Transit (TLS 1.3)                               │        │
│  │  • Database encryption (Transparent Data Encryption)             │        │
│  │  • User data isolation (row-level security)                      │        │
│  │  • PII detection and masking                                     │        │
│  │  • Data retention policies (configurable)                        │        │
│  │  • GDPR/CCPA compliance features                                 │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    AGENT SECURITY                                │        │
│  │                                                                  │        │
│  │  • Prompt injection protection                                   │        │
│  │  • Output sanitization                                           │        │
│  │  • Tool use auditing                                             │        │
│  │  • Rate limiting per user/agent                                  │        │
│  │  • Sandbox network isolation                                     │        │
│  │  • Code execution safety (seccomp, AppArmor)                     │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    AUDIT & COMPLIANCE                            │        │
│  │                                                                  │        │
│  │  • Complete audit trail for all actions                          │        │
│  │  • Immutable log storage                                         │        │
│  │  • SOC 2 Type II compliance                                      │        │
│  │  • Regular security assessments                                  │        │
│  │  • Vulnerability scanning (Trivy, Snyk)                          │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Technology Stack Summary

| Layer | Technology Choices |
|-------|-------------------|
| **Frontend** | Next.js 14, React 18, TypeScript, TailwindCSS, Shadcn/UI |
| **API Layer** | Kong Gateway, GraphQL (Apollo), REST (FastAPI) |
| **Orchestration** | Temporal.io, Apache Airflow, Kubernetes |
| **Agent Framework** | LangGraph, AutoGen, Custom Python agents |
| **LLM Providers** | OpenAI GPT-4o, Anthropic Claude 3.5, Local (Llama 3) |
| **Vector Database** | Qdrant (primary), Pinecone (backup) |
| **Graph Database** | Neo4j Aura |
| **Relational DB** | PostgreSQL 16 with pgvector |
| **Cache** | Redis Cluster 7.x |
| **Search** | Elasticsearch 8.x |
| **Message Queue** | Apache Kafka (Confluent) |
| **Object Storage** | MinIO / AWS S3 |
| **Container Runtime** | Kubernetes 1.29+ (EKS/GKE/AKS) |
| **Monitoring** | Prometheus, Grafana, Jaeger |
| **Secrets** | HashiCorp Vault |
| **CI/CD** | GitHub Actions, ArgoCD |

---

## 📈 Scalability Targets

| Metric | Target | Strategy |
|--------|--------|----------|
| **Concurrent Users** | 10,000+ | Horizontal pod autoscaling, CDN |
| **Queries/Second** | 1,000+ | Caching, read replicas, async processing |
| **Papers Indexed** | 50M+ | Sharded vector DB, incremental updates |
| **User Profiles** | 1M+ | Partitioned PostgreSQL, profile caching |
| **Daily New Content** | 10K+ items | Kafka streaming, batch processing |
| **Sandbox Executions** | 500/day | GPU node pools, spot instances |
| **API Latency (P99)** | <500ms | Edge caching, query optimization |

---

## 🚀 Implementation Roadmap

### Phase 1: Foundation (Months 1-3)
- [ ] Core infrastructure setup (Kubernetes, databases)
- [ ] MCP Gateway implementation
- [ ] Basic agent framework (Orchestrator + 2 acquisition agents)
- [ ] User authentication & basic profiling
- [ ] MVP web interface

### Phase 2: Intelligence (Months 4-6)
- [ ] Full acquisition agent cluster
- [ ] Research & innovation agents
- [ ] Advanced user profiling with temporal preferences
- [ ] Query-driven research workflows
- [ ] Basic proactive recommendations

### Phase 3: Validation (Months 7-9)
- [ ] Sandbox environment implementation
- [ ] Code generation & validation agents
- [ ] Debate mechanism (Bull vs Bear)
- [ ] Dry-run simulation capabilities
- [ ] Advanced personalization engine

### Phase 4: Enterprise (Months 10-12)
- [ ] SSO & enterprise security features
- [ ] Multi-tenancy support
- [ ] Advanced analytics & reporting
- [ ] API marketplace for custom integrations
- [ ] Compliance certifications

---

## 💰 Estimated Infrastructure Costs (Enterprise Scale)

| Component | Monthly Cost (USD) |
|-----------|-------------------|
| Kubernetes Cluster (50 nodes avg) | $15,000 - $25,000 |
| GPU Nodes (Sandbox, spot) | $3,000 - $8,000 |
| Vector Database (50M vectors) | $2,000 - $5,000 |
| PostgreSQL + Redis | $2,000 - $4,000 |
| Elasticsearch Cluster | $3,000 - $6,000 |
| Object Storage (10TB) | $500 - $1,000 |
| LLM API Costs | $10,000 - $30,000 |
| Networking & CDN | $1,000 - $3,000 |
| Monitoring & Logging | $1,000 - $2,000 |
| **Total Estimated** | **$37,500 - $84,000/month** |

---

## 🎯 Key Differentiators

1. **Unified MCP Architecture**: Single protocol for all data sources, enabling easy extension
2. **Learnable Custom Sources**: User-defined sources (Web, RSS, APIs) with quality scores learned from implicit usage signals
3. **Proactive + Reactive Intelligence**: Both push and pull knowledge delivery
4. **Temporal User Profiling**: Distinguishes short-term, medium-term, and long-term interests
5. **Debate-Driven Innovation**: Bull vs Bear agents ensure balanced recommendations
6. **Validated Recommendations**: Every solution is dry-run tested in sandbox before delivery
7. **Enterprise-Grade Security**: Multi-tenant, compliant, auditable

---

This architecture provides a comprehensive, implementable blueprint for building an enterprise-scale deep research platform. The modular design allows for incremental deployment while the MCP-based integration layer ensures extensibility for future data sources and capabilities.

---

# 🔍 Architecture Review & Remediation

## Executive Assessment

The DOVA design is an ambitious enterprise-scale multi-agent research system. While comprehensive in scope, several areas need attention for engineering feasibility and operational viability.

---

## ✅ Design Strengths

### 1. **Architectural Comprehensiveness**
- Well-layered architecture (UI → Orchestration → Agents → MCP → Storage)
- Clear separation of concerns between agent clusters
- Proper data flow diagrams for three core use cases

### 2. **Technology Stack Choices**
- Mature, battle-tested components (Kubernetes, PostgreSQL, Redis, Kafka)
- MCP as a unifying protocol is forward-thinking
- Temporal for workflow orchestration is a solid choice over Airflow for agent coordination

### 3. **Security Design**
- Multi-layer security (RBAC, ABAC, encryption, audit trails)
- Sandbox isolation is well thought-out
- Compliance considerations (SOC 2, GDPR) included

---

## ⚠️ Critical Issues & Remediation

### Issue 1: **Overengineered Initial Scope**

**Problem:** The design requires 7+ databases (PostgreSQL, Redis, Elasticsearch, Neo4j, Qdrant, InfluxDB, S3) from day one.

**Impact:** High operational burden, complex failure modes, expensive.

**Remediation - Phased Database Strategy:**

```
Phase 1 Stack (MVP):
├── PostgreSQL + pgvector (replaces separate vector DB initially)
├── Redis (caching + pub/sub)
└── S3/MinIO (object storage)

Phase 2 Additions (when metrics justify):
├── Qdrant (when >5M vectors)
└── Elasticsearch (when full-text search becomes bottleneck)

Phase 3 Additions (when graph queries prove valuable):
└── Neo4j (after validating graph traversal use cases)
```

---

### Issue 2: **Agent Framework Ambiguity**

**Problem:** Document mentions "LangGraph, AutoGen, Custom Python agents" without clear guidance on when to use each.

**Impact:** Technical debt, inconsistent agent patterns, difficult debugging.

**Remediation - Standardized Agent Base Class:**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class Task:
    id: str
    type: str
    params: dict
    context: dict

@dataclass
class Result:
    success: bool
    data: Any
    error: str | None = None
    metadata: dict = None

class BaseAgent(ABC):
    """All DOVA agents inherit from this base class."""

    def __init__(self, mcp_client: "MCPClient", llm_client: "LLMClient"):
        self.mcp = mcp_client
        self.llm = llm_client
        self.name = self.__class__.__name__

    @abstractmethod
    async def execute(self, task: Task) -> Result:
        """Execute the agent's primary function."""
        pass

    async def call_tool(self, server: str, tool: str, params: dict) -> dict:
        """Unified MCP tool invocation with error handling."""
        return await self.mcp.invoke(server, tool, params)

    async def think(self, prompt: str, **kwargs) -> str:
        """LLM reasoning with configurable provider."""
        return await self.llm.complete(prompt, **kwargs)
```

---

### Issue 3: **Missing Agent Communication Protocol**

**Problem:** A2A (Agent-to-Agent) protocol mentioned but not defined.

**Impact:** Agents can't reliably coordinate on complex tasks.

**Remediation - Explicit Message Schema:**

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from uuid import uuid4

@dataclass
class AgentMessage:
    """Standard message format for inter-agent communication."""

    msg_id: str = field(default_factory=lambda: str(uuid4()))
    from_agent: str = ""
    to_agent: str = ""  # Empty = broadcast
    msg_type: Literal["request", "response", "event"] = "request"
    payload: dict = field(default_factory=dict)
    correlation_id: str = ""  # Links related messages in a conversation
    timestamp: datetime = field(default_factory=datetime.utcnow)
    ttl_seconds: int = 300
    priority: int = 5  # 1=highest, 10=lowest

# Message Bus Implementation
# Use Redis Streams for simplicity, Kafka for scale
# NEVER use direct agent-to-agent RPC calls

class MessageBus:
    async def publish(self, channel: str, message: AgentMessage) -> None:
        """Publish message to channel."""
        pass

    async def subscribe(self, channel: str, handler: Callable) -> None:
        """Subscribe to channel with handler."""
        pass

    async def request_response(
        self,
        to_agent: str,
        payload: dict,
        timeout: float = 30.0
    ) -> AgentMessage:
        """Request-response pattern with timeout."""
        pass
```

---

### Issue 4: **Orchestrator Single Point of Failure**

**Problem:** Master Orchestrator handles intent classification, task decomposition, dispatch, and aggregation all in one component.

**Impact:** Bottleneck, complex testing, hard to scale individual functions.

**Remediation - Split Into Three Services:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    SPLIT ORCHESTRATION LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐                                            │
│  │  Intent Router  │  • Stateless                               │
│  │                 │  • Classifies user intent                  │
│  │                 │  • Enriches with user context              │
│  │                 │  • Routes to appropriate planner           │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │  Task Planner   │  • Stateless                               │
│  │                 │  • Decomposes intent into task DAG         │
│  │                 │  • Estimates resources                     │
│  │                 │  • Assigns priorities                      │
│  └────────┬────────┘                                            │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐                                            │
│  │  Task Executor  │  • Stateful (tracks task progress)         │
│  │                 │  • Dispatches to agent pools               │
│  │                 │  • Aggregates results                      │
│  │                 │  • Handles retries & failures              │
│  └─────────────────┘                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### Issue 5: **User Profiling Privacy Concerns**

**Problem:** Implicit profiling captures extensive behavioral data (scroll depth, copy actions, dwell time) without clear consent mechanism.

**Impact:** GDPR compliance risk, user trust issues.

**Remediation - Privacy-First Profiling:**

```python
@dataclass
class UserProfileConfig:
    """User-controlled profiling preferences."""

    # Always available (required for basic functionality)
    explicit_interests: list[str] = field(default_factory=list)
    preferred_sources: list[str] = field(default_factory=list)
    notification_preferences: dict = field(default_factory=dict)

    # Opt-in only (default: False)
    enable_implicit_profiling: bool = False
    enable_reading_patterns: bool = False
    enable_collaborative_filtering: bool = False

    # Data retention settings
    behavior_data_retention_days: int = 90
    query_history_retention_days: int = 365

class PrivacyCompliantProfiler:
    """GDPR-compliant user profiling."""

    async def update_profile(self, user_id: str, event: dict) -> None:
        config = await self.get_user_config(user_id)

        # Only capture implicit data if user opted in
        if event["type"] == "implicit" and not config.enable_implicit_profiling:
            return

        # Apply differential privacy for aggregate analytics
        if config.enable_collaborative_filtering:
            event = self.apply_differential_privacy(event)

        await self.store_event(user_id, event)

    async def export_user_data(self, user_id: str) -> dict:
        """GDPR Article 20 - Data portability."""
        return await self.get_all_user_data(user_id)

    async def delete_user_data(self, user_id: str) -> None:
        """GDPR Article 17 - Right to erasure."""
        await self.purge_all_user_data(user_id)
```

---

### Issue 6: **Sandbox Cost Model Underestimated**

**Problem:** "500 sandbox executions/day" with GPU (A100) will far exceed stated budget.

**Analysis:**
- Current estimate: $3,000-$8,000/month for GPU
- Reality: A100 spot ≈ $1.50/hr × 4 hrs avg × 500/day = **$90,000/month**

**Remediation - Tiered Sandbox Execution:**

```python
from enum import Enum

class SandboxTier(Enum):
    CPU_BASIC = "cpu_basic"      # 2 vCPU, 4GB RAM, 5 min timeout
    CPU_STANDARD = "cpu_standard" # 4 vCPU, 16GB RAM, 15 min timeout
    GPU_SPOT = "gpu_spot"         # T4 GPU, 16GB RAM, 30 min timeout
    GPU_PREMIUM = "gpu_premium"   # A100 GPU, 80GB RAM, 60 min timeout

@dataclass
class SandboxQuota:
    """Per-user daily sandbox limits."""
    cpu_basic: int = 50
    cpu_standard: int = 20
    gpu_spot: int = 5
    gpu_premium: int = 1  # Requires approval

class SandboxScheduler:
    """Cost-aware sandbox scheduling."""

    TIER_COSTS = {
        SandboxTier.CPU_BASIC: 0.01,      # $/minute
        SandboxTier.CPU_STANDARD: 0.05,
        SandboxTier.GPU_SPOT: 0.50,
        SandboxTier.GPU_PREMIUM: 2.00,
    }

    async def schedule(self, task: SandboxTask, user_id: str) -> SandboxJob:
        # 1. Determine minimum required tier
        tier = self.infer_tier(task)

        # 2. Check user quota
        if not await self.check_quota(user_id, tier):
            raise QuotaExceeded(f"Daily {tier.value} limit reached")

        # 3. Use spot instances when possible
        if tier in [SandboxTier.GPU_SPOT, SandboxTier.GPU_PREMIUM]:
            tier = await self.try_spot_instance(tier)

        # 4. Queue with priority based on user tier
        return await self.queue_job(task, tier, user_id)
```

**Revised Sandbox Budget:**

| Tier | Daily Limit | Avg Duration | Monthly Cost |
|------|-------------|--------------|--------------|
| CPU Basic | 1000 | 3 min | $900 |
| CPU Standard | 200 | 10 min | $3,000 |
| GPU Spot (T4) | 50 | 20 min | $3,000 |
| GPU Premium (A100) | 10 | 30 min | $1,800 |
| **Total** | | | **~$8,700/month** |

---

### Issue 7: **Missing Graceful Degradation**

**Problem:** No fallback strategy when external MCP servers (ArXiv, GitHub, HuggingFace) are unavailable.

**Remediation - Circuit Breaker with Fallbacks:**

```python
from circuitbreaker import circuit
from dataclasses import dataclass
from typing import Optional

@dataclass
class MCPResult:
    data: dict
    source: str  # "live" | "cache" | "fallback"
    stale: bool = False
    error: Optional[str] = None

class ResilientMCPClient:
    """MCP client with circuit breaker and fallback strategies."""

    def __init__(self, cache: CacheClient, fallback_providers: dict):
        self.cache = cache
        self.fallbacks = fallback_providers
        self.circuit_states = {}

    @circuit(failure_threshold=5, recovery_timeout=60)
    async def _call_live(self, server: str, tool: str, params: dict) -> dict:
        """Direct MCP call with circuit breaker."""
        return await self.mcp_client.invoke(server, tool, params)

    async def invoke(
        self,
        server: str,
        tool: str,
        params: dict,
        allow_stale: bool = True
    ) -> MCPResult:
        """Invoke MCP tool with fallback chain."""

        cache_key = self._cache_key(server, tool, params)

        # Try live call first
        try:
            data = await self._call_live(server, tool, params)
            await self.cache.set(cache_key, data, ttl=3600)
            return MCPResult(data=data, source="live")

        except CircuitBreakerError:
            pass  # Fall through to cache/fallback

        # Try cache if allowed
        if allow_stale:
            cached = await self.cache.get(cache_key)
            if cached:
                return MCPResult(data=cached, source="cache", stale=True)

        # Try fallback provider
        if server in self.fallbacks:
            try:
                data = await self.fallbacks[server].invoke(tool, params)
                return MCPResult(data=data, source="fallback")
            except Exception:
                pass

        # Return error with helpful message
        return MCPResult(
            data={},
            source="error",
            error=f"{server} unavailable. Try again in 60s or adjust query."
        )
```

---

### Issue 8: **LLM Provider Lock-in Risk**

**Problem:** Design mentions OpenAI, Anthropic, and local Llama but no abstraction layer. Users should be able to choose their preferred LLM providers based on cost, compliance, or organizational requirements.

**Remediation - Configurable Multi-Provider LLM Router:**

#### 8.1 Provider Configuration Schema

```yaml
# config/llm_providers.yaml
# User-configurable LLM provider settings

llm_providers:
  # ============================================
  # AWS Bedrock Configuration
  # ============================================
  bedrock:
    enabled: true
    priority: 1  # Lower = higher priority
    region: "us-east-1"
    credentials:
      type: "iam_role"  # Options: iam_role, access_key, profile
      role_arn: "${AWS_BEDROCK_ROLE_ARN}"  # For cross-account access
      # Or use access keys (not recommended for production)
      # access_key_id: "${AWS_ACCESS_KEY_ID}"
      # secret_access_key: "${AWS_SECRET_ACCESS_KEY}"
    models:
      reasoning:
        model_id: "anthropic.claude-sonnet-4-20250514-v1:0"
        max_tokens: 4096
        temperature: 0.7
      summarization:
        model_id: "anthropic.claude-3-haiku-20240307-v1:0"
        max_tokens: 2048
        temperature: 0.5
      code_generation:
        model_id: "anthropic.claude-sonnet-4-20250514-v1:0"
        max_tokens: 8192
        temperature: 0.2
      embedding:
        model_id: "amazon.titan-embed-text-v2:0"
        dimensions: 1024
    rate_limits:
      requests_per_minute: 100
      tokens_per_minute: 100000
    cost_per_1k_tokens:
      input: 0.003
      output: 0.015

  # ============================================
  # Anthropic Direct API Configuration
  # ============================================
  anthropic:
    enabled: true
    priority: 2
    api_key: "${ANTHROPIC_API_KEY}"
    base_url: "https://api.anthropic.com"  # Can override for proxy
    models:
      reasoning:
        model_id: "claude-sonnet-4-20250514"
        max_tokens: 4096
        temperature: 0.7
      summarization:
        model_id: "claude-3-haiku-20240307"
        max_tokens: 2048
        temperature: 0.5
      code_generation:
        model_id: "claude-sonnet-4-20250514"
        max_tokens: 8192
        temperature: 0.2
      classification:
        model_id: "claude-3-haiku-20240307"
        max_tokens: 1024
        temperature: 0.3
    rate_limits:
      requests_per_minute: 60
      tokens_per_minute: 80000
    cost_per_1k_tokens:
      input: 0.003
      output: 0.015

  # ============================================
  # OpenAI Configuration
  # ============================================
  openai:
    enabled: true
    priority: 3
    api_key: "${OPENAI_API_KEY}"
    organization_id: "${OPENAI_ORG_ID}"  # Optional
    base_url: "https://api.openai.com/v1"
    models:
      reasoning:
        model_id: "gpt-4o"
        max_tokens: 4096
        temperature: 0.7
      summarization:
        model_id: "gpt-4o-mini"
        max_tokens: 2048
        temperature: 0.5
      code_generation:
        model_id: "gpt-4o"
        max_tokens: 8192
        temperature: 0.2
      embedding:
        model_id: "text-embedding-3-large"
        dimensions: 3072
      classification:
        model_id: "gpt-4o-mini"
        max_tokens: 1024
        temperature: 0.3
    rate_limits:
      requests_per_minute: 60
      tokens_per_minute: 150000
    cost_per_1k_tokens:
      input: 0.0025
      output: 0.01

  # ============================================
  # Azure OpenAI Configuration
  # ============================================
  azure_openai:
    enabled: false  # Enable if using Azure
    priority: 2
    endpoint: "${AZURE_OPENAI_ENDPOINT}"
    api_key: "${AZURE_OPENAI_API_KEY}"
    api_version: "2024-02-15-preview"
    deployment_mappings:
      reasoning: "gpt-4o-deployment"
      summarization: "gpt-4o-mini-deployment"
      code_generation: "gpt-4o-deployment"
      embedding: "text-embedding-deployment"
    rate_limits:
      requests_per_minute: 120
      tokens_per_minute: 200000

  # ============================================
  # Google Vertex AI Configuration
  # ============================================
  vertex_ai:
    enabled: false
    priority: 4
    project_id: "${GCP_PROJECT_ID}"
    location: "us-central1"
    credentials:
      type: "service_account"
      key_file: "${GOOGLE_APPLICATION_CREDENTIALS}"
    models:
      reasoning:
        model_id: "gemini-1.5-pro"
        max_tokens: 4096
      summarization:
        model_id: "gemini-1.5-flash"
        max_tokens: 2048
    rate_limits:
      requests_per_minute: 60
      tokens_per_minute: 100000

  # ============================================
  # Local/Self-Hosted Configuration (vLLM, Ollama, etc.)
  # ============================================
  local:
    enabled: false
    priority: 5
    base_url: "http://localhost:8000/v1"  # vLLM or compatible endpoint
    api_key: "not-needed"  # Some local servers require a dummy key
    models:
      reasoning:
        model_id: "meta-llama/Llama-3.1-70B-Instruct"
        max_tokens: 4096
      summarization:
        model_id: "meta-llama/Llama-3.1-8B-Instruct"
        max_tokens: 2048
      embedding:
        model_id: "BAAI/bge-large-en-v1.5"
        dimensions: 1024
    rate_limits:
      requests_per_minute: 200  # Local = no external rate limits
      tokens_per_minute: 500000
    cost_per_1k_tokens:
      input: 0.0  # Self-hosted = no per-token cost
      output: 0.0

# ============================================
# Task-to-Provider Routing Strategy
# ============================================
routing:
  # Default strategy: priority (use lowest priority number first)
  # Options: priority, cost, latency, round_robin
  strategy: "priority"

  # Task-specific overrides
  task_overrides:
    embedding:
      # Always use same provider for embeddings (consistency)
      strategy: "fixed"
      fixed_provider: "bedrock"
    code_generation:
      # Prefer lowest latency for interactive coding
      strategy: "latency"

  # Fallback behavior
  fallback:
    enabled: true
    max_retries: 3
    retry_delay_seconds: 2
    exclude_providers: []  # Providers to never fallback to

# ============================================
# User-Level Provider Preferences
# ============================================
user_preferences:
  # Allow users to override default provider selection
  allow_user_override: true

  # Available options users can choose from
  selectable_providers:
    - "bedrock"
    - "anthropic"
    - "openai"
    - "azure_openai"

  # Default for new users
  default_provider: "bedrock"
```

#### 8.2 User Preferences Schema

```yaml
# Per-user LLM preferences (stored in user profile)
# Example: user_preferences/user_123.yaml

user_llm_preferences:
  # User's preferred provider (overrides system default)
  preferred_provider: "bedrock"

  # Fallback chain if preferred is unavailable
  fallback_providers:
    - "anthropic"
    - "openai"

  # Task-specific preferences
  task_preferences:
    code_generation:
      provider: "anthropic"
      model_override: "claude-sonnet-4-20250514"
      temperature: 0.1  # User prefers more deterministic code
    reasoning:
      provider: "bedrock"
      # Use default model

  # Cost controls
  cost_limits:
    daily_budget_usd: 10.00
    alert_threshold_percent: 80

  # Compliance requirements
  compliance:
    # Only use providers that meet these requirements
    require_soc2: true
    require_gdpr: true
    allowed_regions:
      - "us-east-1"
      - "eu-west-1"
    # This would exclude some providers automatically
```

#### 8.3 LLM Router Implementation

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional
import yaml
import boto3
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

class TaskType(Enum):
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    CODE_GENERATION = "code_generation"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"

class RoutingStrategy(Enum):
    PRIORITY = "priority"
    COST = "cost"
    LATENCY = "latency"
    ROUND_ROBIN = "round_robin"
    FIXED = "fixed"

@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str
    usage: dict
    latency_ms: float
    cost_usd: float

@dataclass
class ProviderConfig:
    name: str
    enabled: bool
    priority: int
    models: dict
    rate_limits: dict
    cost_per_1k_tokens: dict
    credentials: dict = field(default_factory=dict)

# ============================================
# Provider Interface (Abstract Base)
# ============================================
class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = config.name

    @abstractmethod
    async def complete(
        self,
        task_type: TaskType,
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        pass

    @abstractmethod
    async def stream(
        self,
        task_type: TaskType,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[str]:
        pass

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass

# ============================================
# AWS Bedrock Provider
# ============================================
class BedrockProvider(LLMProvider):
    """AWS Bedrock LLM provider implementation."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = self._create_client()

    def _create_client(self):
        """Create Bedrock client with appropriate credentials."""
        creds = self.config.credentials

        if creds.get("type") == "iam_role":
            # Use IAM role (recommended for EC2/EKS)
            session = boto3.Session()
            if creds.get("role_arn"):
                # Assume cross-account role if specified
                sts = session.client("sts")
                assumed = sts.assume_role(
                    RoleArn=creds["role_arn"],
                    RoleSessionName="dova-llm-router"
                )
                session = boto3.Session(
                    aws_access_key_id=assumed["Credentials"]["AccessKeyId"],
                    aws_secret_access_key=assumed["Credentials"]["SecretAccessKey"],
                    aws_session_token=assumed["Credentials"]["SessionToken"]
                )
        else:
            # Use explicit credentials (dev/testing only)
            session = boto3.Session(
                aws_access_key_id=creds.get("access_key_id"),
                aws_secret_access_key=creds.get("secret_access_key"),
                region_name=self.config.credentials.get("region", "us-east-1")
            )

        return session.client(
            "bedrock-runtime",
            region_name=self.config.credentials.get("region", "us-east-1")
        )

    async def complete(
        self,
        task_type: TaskType,
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        import time
        import json

        model_config = self.config.models.get(task_type.value, {})
        model_id = model_config.get("model_id")

        start_time = time.time()

        # Bedrock uses different request formats per model family
        if "anthropic" in model_id:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": kwargs.get("max_tokens", model_config.get("max_tokens", 4096)),
                "temperature": kwargs.get("temperature", model_config.get("temperature", 0.7)),
                "messages": [{"role": "user", "content": prompt}]
            }
        elif "amazon.titan" in model_id:
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": kwargs.get("max_tokens", model_config.get("max_tokens", 4096)),
                    "temperature": kwargs.get("temperature", model_config.get("temperature", 0.7)),
                }
            }
        else:
            # Generic format for other models
            body = {
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", model_config.get("max_tokens", 4096)),
                "temperature": kwargs.get("temperature", model_config.get("temperature", 0.7)),
            }

        response = self.client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )

        response_body = json.loads(response["body"].read())
        latency_ms = (time.time() - start_time) * 1000

        # Parse response based on model family
        if "anthropic" in model_id:
            content = response_body["content"][0]["text"]
            usage = response_body.get("usage", {})
        elif "amazon.titan" in model_id:
            content = response_body["results"][0]["outputText"]
            usage = {"input_tokens": 0, "output_tokens": 0}  # Titan doesn't return usage
        else:
            content = response_body.get("completion", response_body.get("generated_text", ""))
            usage = {}

        # Calculate cost
        input_tokens = usage.get("input_tokens", len(prompt) // 4)
        output_tokens = usage.get("output_tokens", len(content) // 4)
        cost = (
            (input_tokens / 1000) * self.config.cost_per_1k_tokens.get("input", 0) +
            (output_tokens / 1000) * self.config.cost_per_1k_tokens.get("output", 0)
        )

        return LLMResponse(
            content=content,
            provider="bedrock",
            model=model_id,
            usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
            latency_ms=latency_ms,
            cost_usd=cost
        )

    async def stream(
        self,
        task_type: TaskType,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream response from Bedrock."""
        import json

        model_config = self.config.models.get(task_type.value, {})
        model_id = model_config.get("model_id")

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": kwargs.get("max_tokens", model_config.get("max_tokens", 4096)),
            "temperature": kwargs.get("temperature", model_config.get("temperature", 0.7)),
            "messages": [{"role": "user", "content": prompt}]
        }

        response = self.client.invoke_model_with_response_stream(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json"
        )

        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk["type"] == "content_block_delta":
                yield chunk["delta"].get("text", "")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Bedrock Titan."""
        import json

        model_config = self.config.models.get("embedding", {})
        model_id = model_config.get("model_id", "amazon.titan-embed-text-v2:0")

        embeddings = []
        for text in texts:
            response = self.client.invoke_model(
                modelId=model_id,
                body=json.dumps({"inputText": text}),
                contentType="application/json"
            )
            result = json.loads(response["body"].read())
            embeddings.append(result["embedding"])

        return embeddings

    async def health_check(self) -> bool:
        try:
            # Simple health check with minimal tokens
            await self.complete(TaskType.CLASSIFICATION, "test", max_tokens=5)
            return True
        except Exception:
            return False

# ============================================
# Anthropic Direct Provider
# ============================================
class AnthropicProvider(LLMProvider):
    """Anthropic direct API provider."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = AsyncAnthropic(
            api_key=config.credentials.get("api_key"),
            base_url=config.credentials.get("base_url")
        )

    async def complete(
        self,
        task_type: TaskType,
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        import time

        model_config = self.config.models.get(task_type.value, {})
        start_time = time.time()

        response = await self.client.messages.create(
            model=kwargs.get("model", model_config.get("model_id")),
            max_tokens=kwargs.get("max_tokens", model_config.get("max_tokens", 4096)),
            temperature=kwargs.get("temperature", model_config.get("temperature", 0.7)),
            messages=[{"role": "user", "content": prompt}]
        )

        latency_ms = (time.time() - start_time) * 1000
        content = response.content[0].text

        cost = (
            (response.usage.input_tokens / 1000) * self.config.cost_per_1k_tokens.get("input", 0) +
            (response.usage.output_tokens / 1000) * self.config.cost_per_1k_tokens.get("output", 0)
        )

        return LLMResponse(
            content=content,
            provider="anthropic",
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            },
            latency_ms=latency_ms,
            cost_usd=cost
        )

    async def stream(
        self,
        task_type: TaskType,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[str]:
        model_config = self.config.models.get(task_type.value, {})

        async with self.client.messages.stream(
            model=kwargs.get("model", model_config.get("model_id")),
            max_tokens=kwargs.get("max_tokens", model_config.get("max_tokens", 4096)),
            temperature=kwargs.get("temperature", model_config.get("temperature", 0.7)),
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Anthropic does not provide embeddings API")

    async def health_check(self) -> bool:
        try:
            await self.complete(TaskType.CLASSIFICATION, "test", max_tokens=5)
            return True
        except Exception:
            return False

# ============================================
# OpenAI Provider
# ============================================
class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=config.credentials.get("api_key"),
            organization=config.credentials.get("organization_id"),
            base_url=config.credentials.get("base_url")
        )

    async def complete(
        self,
        task_type: TaskType,
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        import time

        model_config = self.config.models.get(task_type.value, {})
        start_time = time.time()

        response = await self.client.chat.completions.create(
            model=kwargs.get("model", model_config.get("model_id")),
            max_tokens=kwargs.get("max_tokens", model_config.get("max_tokens", 4096)),
            temperature=kwargs.get("temperature", model_config.get("temperature", 0.7)),
            messages=[{"role": "user", "content": prompt}]
        )

        latency_ms = (time.time() - start_time) * 1000
        content = response.choices[0].message.content

        cost = (
            (response.usage.prompt_tokens / 1000) * self.config.cost_per_1k_tokens.get("input", 0) +
            (response.usage.completion_tokens / 1000) * self.config.cost_per_1k_tokens.get("output", 0)
        )

        return LLMResponse(
            content=content,
            provider="openai",
            model=response.model,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            },
            latency_ms=latency_ms,
            cost_usd=cost
        )

    async def stream(
        self,
        task_type: TaskType,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[str]:
        model_config = self.config.models.get(task_type.value, {})

        stream = await self.client.chat.completions.create(
            model=kwargs.get("model", model_config.get("model_id")),
            max_tokens=kwargs.get("max_tokens", model_config.get("max_tokens", 4096)),
            temperature=kwargs.get("temperature", model_config.get("temperature", 0.7)),
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def embed(self, texts: list[str]) -> list[list[float]]:
        model_config = self.config.models.get("embedding", {})

        response = await self.client.embeddings.create(
            model=model_config.get("model_id", "text-embedding-3-large"),
            input=texts
        )

        return [item.embedding for item in response.data]

    async def health_check(self) -> bool:
        try:
            await self.complete(TaskType.CLASSIFICATION, "test", max_tokens=5)
            return True
        except Exception:
            return False

# ============================================
# Main LLM Router
# ============================================
class LLMRouter:
    """
    Configurable LLM router with multi-provider support.
    Routes requests based on user preferences, task type, and provider availability.
    """

    PROVIDER_CLASSES = {
        "bedrock": BedrockProvider,
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        # "azure_openai": AzureOpenAIProvider,
        # "vertex_ai": VertexAIProvider,
        # "local": LocalProvider,
    }

    def __init__(self, config_path: str = "config/llm_providers.yaml"):
        self.config = self._load_config(config_path)
        self.providers: dict[str, LLMProvider] = {}
        self.health_status: dict[str, bool] = {}
        self._init_providers()

    def _load_config(self, path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def _init_providers(self):
        """Initialize enabled providers."""
        for name, provider_config in self.config.get("llm_providers", {}).items():
            if provider_config.get("enabled", False):
                provider_class = self.PROVIDER_CLASSES.get(name)
                if provider_class:
                    config = ProviderConfig(
                        name=name,
                        enabled=True,
                        priority=provider_config.get("priority", 99),
                        models=provider_config.get("models", {}),
                        rate_limits=provider_config.get("rate_limits", {}),
                        cost_per_1k_tokens=provider_config.get("cost_per_1k_tokens", {}),
                        credentials=provider_config
                    )
                    self.providers[name] = provider_class(config)
                    self.health_status[name] = True

    def _get_provider_order(
        self,
        task_type: TaskType,
        user_preferences: Optional[dict] = None,
        strategy: Optional[RoutingStrategy] = None
    ) -> list[str]:
        """Determine provider order based on strategy and preferences."""

        # Check for task-specific strategy override
        routing_config = self.config.get("routing", {})
        task_overrides = routing_config.get("task_overrides", {})

        if task_type.value in task_overrides:
            override = task_overrides[task_type.value]
            strategy = RoutingStrategy(override.get("strategy", "priority"))
            if strategy == RoutingStrategy.FIXED:
                fixed = override.get("fixed_provider")
                if fixed and fixed in self.providers:
                    return [fixed]

        # Use default strategy if not specified
        if not strategy:
            strategy = RoutingStrategy(routing_config.get("strategy", "priority"))

        # Apply user preferences if provided
        if user_preferences:
            preferred = user_preferences.get("preferred_provider")
            if preferred and preferred in self.providers:
                fallbacks = user_preferences.get("fallback_providers", [])
                return [preferred] + [p for p in fallbacks if p in self.providers]

        # Sort providers by strategy
        available = [
            (name, p) for name, p in self.providers.items()
            if self.health_status.get(name, False)
        ]

        if strategy == RoutingStrategy.PRIORITY:
            available.sort(key=lambda x: x[1].config.priority)
        elif strategy == RoutingStrategy.COST:
            available.sort(key=lambda x: x[1].config.cost_per_1k_tokens.get("output", 999))
        elif strategy == RoutingStrategy.ROUND_ROBIN:
            # Implement round-robin state externally
            pass

        return [name for name, _ in available]

    async def complete(
        self,
        task_type: TaskType,
        prompt: str,
        user_id: Optional[str] = None,
        user_preferences: Optional[dict] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Complete prompt with automatic provider selection and fallback.

        Args:
            task_type: Type of task for model selection
            prompt: The prompt to complete
            user_id: Optional user ID for preference lookup
            user_preferences: Optional direct preferences dict
            **kwargs: Additional arguments passed to provider

        Returns:
            LLMResponse with content and metadata
        """
        provider_order = self._get_provider_order(task_type, user_preferences)
        last_error = None

        for provider_name in provider_order:
            provider = self.providers.get(provider_name)
            if not provider:
                continue

            try:
                response = await provider.complete(task_type, prompt, **kwargs)
                return response

            except Exception as e:
                last_error = e
                self.health_status[provider_name] = False
                # Schedule health check recovery
                continue

        raise AllProvidersUnavailable(
            f"All providers failed for {task_type.value}: {last_error}"
        )

    async def stream(
        self,
        task_type: TaskType,
        prompt: str,
        user_preferences: Optional[dict] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream completion with automatic fallback."""
        provider_order = self._get_provider_order(task_type, user_preferences)

        for provider_name in provider_order:
            provider = self.providers.get(provider_name)
            if not provider or not self.health_status.get(provider_name, False):
                continue

            try:
                async for chunk in provider.stream(task_type, prompt, **kwargs):
                    yield chunk
                return
            except Exception:
                self.health_status[provider_name] = False
                continue

        raise AllProvidersUnavailable(f"All providers failed for {task_type.value}")

    async def embed(
        self,
        texts: list[str],
        user_preferences: Optional[dict] = None
    ) -> list[list[float]]:
        """Generate embeddings with automatic fallback."""
        provider_order = self._get_provider_order(TaskType.EMBEDDING, user_preferences)

        for provider_name in provider_order:
            provider = self.providers.get(provider_name)
            if not provider:
                continue

            try:
                return await provider.embed(texts)
            except NotImplementedError:
                continue
            except Exception:
                self.health_status[provider_name] = False
                continue

        raise AllProvidersUnavailable("No embedding provider available")

    async def refresh_health_status(self):
        """Refresh health status for all providers."""
        for name, provider in self.providers.items():
            self.health_status[name] = await provider.health_check()

class AllProvidersUnavailable(Exception):
    """Raised when all LLM providers are unavailable."""
    pass
```

#### 8.4 User Provider Selection API

```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])

class LLMPreferences(BaseModel):
    preferred_provider: str
    fallback_providers: list[str] = []
    task_preferences: dict = {}
    cost_limits: dict = {}

@router.get("/llm-providers")
async def list_available_providers():
    """List all available LLM providers for user selection."""
    return {
        "providers": [
            {
                "id": "bedrock",
                "name": "AWS Bedrock",
                "description": "AWS-managed Claude, Llama, and other models",
                "features": ["SOC2", "HIPAA", "Private VPC"],
                "regions": ["us-east-1", "us-west-2", "eu-west-1"]
            },
            {
                "id": "anthropic",
                "name": "Anthropic Direct",
                "description": "Direct API access to Claude models",
                "features": ["Latest models", "Fast updates"],
                "regions": ["global"]
            },
            {
                "id": "openai",
                "name": "OpenAI",
                "description": "GPT-4o and other OpenAI models",
                "features": ["GPT-4o", "Embeddings", "Vision"],
                "regions": ["global"]
            },
            {
                "id": "azure_openai",
                "name": "Azure OpenAI",
                "description": "Microsoft Azure-hosted OpenAI models",
                "features": ["Enterprise SLA", "Private endpoints", "GDPR"],
                "regions": ["Multiple Azure regions"]
            }
        ]
    }

@router.get("/llm-preferences")
async def get_user_llm_preferences(user_id: str = Depends(get_current_user)):
    """Get current user's LLM provider preferences."""
    prefs = await db.get_user_preferences(user_id)
    return prefs.get("llm_preferences", {})

@router.put("/llm-preferences")
async def update_user_llm_preferences(
    preferences: LLMPreferences,
    user_id: str = Depends(get_current_user)
):
    """Update user's LLM provider preferences."""
    # Validate provider is available
    available = {"bedrock", "anthropic", "openai", "azure_openai"}
    if preferences.preferred_provider not in available:
        raise HTTPException(400, f"Invalid provider: {preferences.preferred_provider}")

    for provider in preferences.fallback_providers:
        if provider not in available:
            raise HTTPException(400, f"Invalid fallback provider: {provider}")

    await db.update_user_preferences(user_id, {"llm_preferences": preferences.dict()})
    return {"status": "updated"}
```

#### 8.5 Provider Comparison Matrix

| Provider | Compliance | Latency | Cost | Best For |
|----------|------------|---------|------|----------|
| **AWS Bedrock** | SOC2, HIPAA, FedRAMP | Medium | Medium | Enterprise, regulated industries |
| **Anthropic Direct** | SOC2 | Low | Medium | Latest features, research |
| **OpenAI** | SOC2 | Low | Low-Medium | General purpose, embeddings |
| **Azure OpenAI** | SOC2, HIPAA, GDPR | Medium | Medium | Microsoft ecosystem, EU data residency |
| **Google Vertex AI** | SOC2, HIPAA | Medium | Medium | GCP ecosystem, Gemini models |
| **Local (vLLM)** | N/A (self-hosted) | Very Low | Low (infra only) | Air-gapped, full control |

---

## 🔧 Additional Required Components

### 1. Rate Limiting Configuration

```yaml
# config/rate_limits.yaml
rate_limits:
  # Per-user limits
  user:
    queries_per_minute: 20
    queries_per_day: 500
    sandbox_executions_per_day: 10

  # Per-agent limits (prevent runaway costs)
  agent:
    llm_calls_per_minute: 100
    llm_calls_per_task: 20
    mcp_calls_per_minute: 50

  # Per-MCP-server limits (respect external rate limits)
  mcp_servers:
    arxiv:
      requests_per_minute: 30
      requests_per_day: 5000
    github:
      requests_per_minute: 60
      requests_per_hour: 1000
    huggingface:
      requests_per_minute: 30
```

### 2. Retry & Backoff Policy

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class RetryConfig:
    max_retries: int
    backoff: Literal["none", "linear", "exponential"]
    initial_delay: float  # seconds
    max_delay: float  # seconds
    retryable_errors: list[str]

RETRY_CONFIGS = {
    "mcp_calls": RetryConfig(
        max_retries=3,
        backoff="exponential",
        initial_delay=1.0,
        max_delay=30.0,
        retryable_errors=["timeout", "rate_limited", "server_error"]
    ),
    "llm_calls": RetryConfig(
        max_retries=2,
        backoff="linear",
        initial_delay=2.0,
        max_delay=10.0,
        retryable_errors=["timeout", "rate_limited", "overloaded"]
    ),
    "sandbox_execution": RetryConfig(
        max_retries=1,
        backoff="none",
        initial_delay=0,
        max_delay=0,
        retryable_errors=["infrastructure_error"]  # NOT code errors
    ),
    "database": RetryConfig(
        max_retries=3,
        backoff="exponential",
        initial_delay=0.5,
        max_delay=5.0,
        retryable_errors=["connection_error", "timeout"]
    )
}
```

### 3. Health Check Endpoints

```python
from fastapi import FastAPI, Response
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

app = FastAPI()

@app.get("/health/live")
async def liveness():
    """Kubernetes liveness probe - is the process running?"""
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness():
    """Kubernetes readiness probe - can we serve requests?"""
    checks = {
        "database": await check_db_connection(),
        "redis": await check_redis_connection(),
        "message_bus": await check_message_bus(),
    }

    all_healthy = all(c["healthy"] for c in checks.values())
    status_code = 200 if all_healthy else 503

    return Response(
        content=json.dumps({"ready": all_healthy, "checks": checks}),
        status_code=status_code
    )

@app.get("/health/deps")
async def dependency_health():
    """Detailed dependency status for debugging."""
    return {
        "database": await detailed_db_check(),
        "redis": await detailed_redis_check(),
        "mcp_servers": {
            "arxiv": await check_mcp_server("arxiv"),
            "github": await check_mcp_server("github"),
            "huggingface": await check_mcp_server("huggingface"),
        },
        "llm_providers": {
            "anthropic": await check_llm_provider("anthropic"),
            "openai": await check_llm_provider("openai"),
        }
    }
```

### 4. Feature Flags

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class FeatureFlag:
    name: str
    enabled: bool
    rollout_percent: int  # 0-100
    allowed_users: Optional[list[str]] = None  # Override for specific users
    metadata: dict = None

FEATURE_FLAGS = {
    "proactive_recommendations": FeatureFlag(
        name="proactive_recommendations",
        enabled=False,
        rollout_percent=0,
        metadata={"phase": "beta", "owner": "recommendations-team"}
    ),
    "sandbox_gpu": FeatureFlag(
        name="sandbox_gpu",
        enabled=True,
        rollout_percent=20,
        allowed_users=["power-users-group"],
        metadata={"phase": "limited-beta"}
    ),
    "debate_agents": FeatureFlag(
        name="debate_agents",
        enabled=False,
        rollout_percent=0,
        metadata={"phase": "development"}
    ),
    "neo4j_graph_queries": FeatureFlag(
        name="neo4j_graph_queries",
        enabled=False,
        rollout_percent=10,
        metadata={"phase": "validation"}
    )
}

class FeatureFlagService:
    async def is_enabled(self, flag_name: str, user_id: str) -> bool:
        flag = FEATURE_FLAGS.get(flag_name)
        if not flag or not flag.enabled:
            return False

        # Check user override
        if flag.allowed_users and user_id in flag.allowed_users:
            return True

        # Check rollout percentage
        user_bucket = hash(f"{flag_name}:{user_id}") % 100
        return user_bucket < flag.rollout_percent
```

---

## 📋 Revised Implementation Roadmap

### Phase 0: Spike & Validation (2 weeks)
- [ ] MCP integration POC with ArXiv + GitHub
- [ ] Agent framework decision (validate BaseAgent pattern)
- [ ] LLM provider abstraction POC
- [ ] Database schema draft

**Exit Criteria:** Working query → MCP → LLM → response pipeline

### Phase 1: Core Platform (8 weeks)
- [ ] PostgreSQL + pgvector setup
- [ ] Redis for caching + message bus
- [ ] Single query flow end-to-end
- [ ] Basic web UI (query input + results display)
- [ ] User authentication (OAuth)
- [ ] Health check endpoints

**Exit Criteria:** User can login, submit query, receive results from ArXiv/GitHub

### Phase 2: Acquisition Agents (6 weeks)
- [ ] ArXiv acquisition agent
- [ ] GitHub acquisition agent
- [ ] HuggingFace acquisition agent
- [ ] Result aggregation + deduplication
- [ ] Basic caching layer

**Exit Criteria:** Multi-source queries return merged, deduplicated results

### Phase 3: User Profiling (4 weeks)
- [ ] Explicit preference storage
- [ ] Query history tracking
- [ ] Basic personalization (filter by interests)
- [ ] Privacy controls UI

**Exit Criteria:** Users can set interests, queries prioritize matching content

### Phase 4: Proactive Intelligence (6 weeks)
- [ ] Background content monitoring
- [ ] User-content matching algorithm
- [ ] Notification system (email + in-app)
- [ ] Frequency capping

**Exit Criteria:** Users receive relevant paper alerts without querying

### Phase 5: Sandbox Environment (6 weeks)
- [ ] Kubernetes sandbox namespace
- [ ] CPU-only execution tier
- [ ] Code generation agent (basic)
- [ ] Result validation framework
- [ ] User quota management

**Exit Criteria:** Users can run generated code snippets safely

### Phase 6: Enterprise Features (8 weeks)
- [ ] SSO integration (SAML/OIDC)
- [ ] Multi-tenant data isolation
- [ ] Admin dashboard
- [ ] Usage analytics
- [ ] Compliance audit logging

**Exit Criteria:** Enterprise pilot deployment ready

---

## 💰 Revised Cost Estimates

### MVP Phase (Months 1-3)

| Component | Monthly Cost |
|-----------|-------------|
| Kubernetes (3-5 nodes) | $1,500 - $3,000 |
| PostgreSQL (managed) | $200 - $500 |
| Redis (managed) | $100 - $200 |
| S3/MinIO (100GB) | $50 - $100 |
| LLM APIs (dev usage) | $500 - $2,000 |
| Monitoring (basic) | $100 - $200 |
| **Total MVP** | **$2,450 - $6,000/month** |

### Growth Phase (Months 4-6)

| Component | Monthly Cost |
|-----------|-------------|
| Kubernetes (8-12 nodes) | $5,000 - $8,000 |
| PostgreSQL + Read Replicas | $500 - $1,000 |
| Redis Cluster | $300 - $500 |
| Qdrant (5M vectors) | $500 - $1,000 |
| S3/MinIO (500GB) | $100 - $300 |
| LLM APIs (production) | $5,000 - $10,000 |
| Monitoring (full stack) | $300 - $500 |
| **Total Growth** | **$11,700 - $21,300/month** |

### Scale Phase (Months 7+)

| Component | Monthly Cost |
|-----------|-------------|
| Kubernetes (20-30 nodes) | $12,000 - $20,000 |
| PostgreSQL (HA) | $1,500 - $3,000 |
| Redis Cluster (HA) | $800 - $1,500 |
| Qdrant (20M vectors) | $2,000 - $4,000 |
| Elasticsearch (3 nodes) | $2,000 - $4,000 |
| S3/MinIO (2TB) | $300 - $500 |
| GPU Sandbox Pool | $5,000 - $10,000 |
| LLM APIs (scale) | $15,000 - $30,000 |
| Neo4j (if validated) | $1,000 - $2,000 |
| Monitoring + Alerting | $500 - $1,000 |
| **Total Scale** | **$40,100 - $76,000/month** |

---

## 📊 Final Assessment Matrix

| Criterion | Before Remediation | After Remediation |
|-----------|-------------------|-------------------|
| **Comprehensive** | ✅ Strong | ✅ Strong |
| **Engineering Feasibility** | ⚠️ Overengineered | ✅ Phased approach |
| **Operational Viability** | ⚠️ Missing pieces | ✅ Complete |
| **Cost Predictability** | ❌ Underestimated | ✅ Realistic |
| **Privacy Compliance** | ⚠️ Risks | ✅ GDPR-ready |
| **Resilience** | ❌ No fallbacks | ✅ Circuit breakers |

---

## 🎯 Key Success Metrics

Track these metrics to validate architecture decisions:

| Metric | Target (MVP) | Target (Scale) |
|--------|-------------|----------------|
| Query latency (P50) | < 3s | < 2s |
| Query latency (P99) | < 10s | < 5s |
| MCP availability | > 95% | > 99% |
| User satisfaction (NPS) | > 30 | > 50 |
| Cost per query | < $0.10 | < $0.05 |
| Sandbox success rate | > 80% | > 90% |

---

**Recommendation:** Start with "DOVA Lite" - a minimal implementation that delivers value in 3 months, then iterate toward the full architecture based on validated user needs.

---

# 🚀 Implementation with AWS Strands Agents + Amazon Bedrock AgentCore

## Executive Summary

**Verdict: ✅ HIGHLY FEASIBLE** - DOVA can be implemented using **Strands Agents SDK** and **Amazon Bedrock AgentCore** as the core framework, with significant reduction in custom development effort (60%+) and infrastructure costs (50%+).

---

## Technology Overview

### Strands Agents SDK

| Attribute | Details |
|-----------|---------|
| **Source** | AWS Open Source (Apache 2.0) |
| **GitHub** | [strands-agents/sdk-python](https://github.com/strands-agents/sdk-python) (~5,000 ⭐) |
| **Maturity** | Production-ready (used internally at AWS) |
| **Key Features** | Native MCP support, multi-agent patterns, model-agnostic |
| **Documentation** | [strandsagents.com](https://strandsagents.com) |

**Core Capabilities:**
- **Model Agnostic**: Amazon Bedrock, Anthropic, OpenAI, Ollama, Gemini, LiteLLM, and custom providers
- **Native MCP Support**: First-class integration with Model Context Protocol servers
- **Multi-Agent Patterns**: Swarm intelligence, Graph-based coordination, Agents-as-Tools
- **Built-in Tools**: 40+ production-ready tools (web search, file ops, AWS integration, etc.)
- **Observability**: OpenTelemetry + LangFuse integration out of the box

### Amazon Bedrock AgentCore

| Component | Purpose | DOVA Mapping |
|-----------|---------|--------------|
| **AgentCore Runtime** | Serverless agent deployment with auto-scaling | Replaces Kubernetes + custom orchestration |
| **AgentCore Memory** | Persistent memory across sessions | Replaces custom user profiling database |
| **AgentCore Gateway** | Authenticated tool integration via Lambda | Replaces custom MCP Gateway |
| **Code Interpreter** | Secure sandbox for code execution | Replaces custom sandbox environment |

**Memory Strategies (Built-in):**
1. **User Preferences** - Stores user-specific preferences and settings
2. **Semantic Facts** - Stores factual information and knowledge
3. **Session Summaries** - Maintains conversation context and summaries

---

## DOVA Component Mapping

### ✅ Components Directly Supported (Low/Very Low Effort)

| DOVA Component | Strands/AgentCore Solution | Effort | Notes |
|----------------|---------------------------|--------|-------|
| **Master Orchestrator** | Strands Agent with Graph/Swarm patterns | Low | Use `agent_graph` or `swarm` tools |
| **MCP Integration Layer** | Native `MCPClient` in Strands SDK | Very Low | First-class support |
| **ArXiv Agent** | Strands + ArXiv MCP server | Very Low | MCP server exists |
| **GitHub Agent** | Strands + GitHub MCP server | Very Low | MCP server exists |
| **HuggingFace Agent** | Strands + HuggingFace MCP server | Very Low | MCP server exists |
| **Web Search Agent** | `tavily_search`, `exa_search` built-in tools | Very Low | Production-ready |
| **Code Sandbox** | AgentCore Code Interpreter | Very Low | Managed service |
| **User Memory** | AgentCore Memory (3 strategies) | Low | Managed service |
| **Agent-to-Agent Comm** | A2A protocol + `a2a_client` tool | Low | Built-in support |
| **LLM Abstraction** | Built-in multi-provider support | Very Low | Bedrock, Anthropic, OpenAI, etc. |
| **Observability** | OpenTelemetry + LangFuse | Low | Built-in integration |
| **HTTP/API Tools** | `http_request`, `use_aws` tools | Very Low | Built-in |
| **File Operations** | `file_read`, `file_write`, `editor` tools | Very Low | Built-in |

### ⚠️ Components Requiring Custom Development (Medium Effort)

| DOVA Component | Gap | Solution | Effort |
|----------------|-----|----------|--------|
| **Temporal User Profiling** | AgentCore Memory lacks decay functions | Custom memory wrapper with temporal logic | Medium |
| **Debate Agents (Bull/Bear)** | Not a built-in pattern | Custom multi-agent using Swarm with opposing prompts | Medium |
| **Proactive Recommendations** | Event-driven push not built-in | Lambda + EventBridge + Strands agents | Medium |
| **Literature Synthesis** | Domain-specific logic | Custom agent with RAG pattern using `retrieve` tool | Medium |
| **Enterprise SSO** | AgentCore uses Cognito | FAST template supports SAML/OIDC federation | Low |

### ❌ Components Replaced by Managed Services

| Original DOVA Component | Replaced By | Benefit |
|------------------------|-------------|---------|
| Kubernetes Cluster | AgentCore Runtime | No cluster management |
| Temporal.io / Airflow | AgentCore Runtime + Step Functions | Serverless orchestration |
| Custom Vector DB (Qdrant) | AgentCore Memory + Bedrock KB | Managed embeddings |
| Custom Sandbox (Docker) | AgentCore Code Interpreter | Secure, managed execution |
| Kong/Nginx API Gateway | AgentCore Gateway | Integrated auth |
| Custom User Profile DB | AgentCore Memory | 3 built-in strategies |

---

## Revised Architecture with Strands + AgentCore

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         DOVA on Strands Agents + Amazon Bedrock AgentCore                │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                              USER INTERFACE LAYER                                │    │
│  │                                                                                  │    │
│  │   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │    │
│  │   │  React Frontend │    │  AgentCore      │    │  AWS Cognito    │            │    │
│  │   │  (FAST Template)│◄──►│  Gateway        │◄──►│  (SSO/OAuth)    │            │    │
│  │   │  Next.js + htmx │    │  (API + Auth)   │    │  SAML/OIDC      │            │    │
│  │   └─────────────────┘    └─────────────────┘    └─────────────────┘            │    │
│  │                                                                                  │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                          │                                               │
│                                          ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                         AGENTCORE RUNTIME (Serverless)                           │    │
│  │                                                                                  │    │
│  │   ┌───────────────────────────────────────────────────────────────────────┐     │    │
│  │   │                      STRANDS AGENTS LAYER                              │     │    │
│  │   │                                                                        │     │    │
│  │   │   ┌─────────────────────────────────────────────────────────────┐     │     │    │
│  │   │   │                 MASTER ORCHESTRATOR AGENT                    │     │     │    │
│  │   │   │                 (Graph-based coordination)                   │     │     │    │
│  │   │   │                                                              │     │     │    │
│  │   │   │   • Intent classification via LLM                           │     │     │    │
│  │   │   │   • Task decomposition using agent_graph tool               │     │     │    │
│  │   │   │   • Parallel execution via batch tool                       │     │     │    │
│  │   │   │   • Result synthesis and personalization                    │     │     │    │
│  │   │   └─────────────────────────────────────────────────────────────┘     │     │    │
│  │   │                              │                                         │     │    │
│  │   │              ┌───────────────┼───────────────┐                        │     │    │
│  │   │              ▼               ▼               ▼                        │     │    │
│  │   │   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                 │     │    │
│  │   │   │ ACQUISITION  │ │ RESEARCH &   │ │ VALIDATION   │                 │     │    │
│  │   │   │ AGENT SWARM  │ │ INNOVATION   │ │ AGENT        │                 │     │    │
│  │   │   │              │ │ AGENTS       │ │              │                 │     │    │
│  │   │   │ • ArXiv      │ │ • Synthesis  │ │ • Code       │                 │     │    │
│  │   │   │ • GitHub     │ │ • Hypothesis │ │   Interpreter│                 │     │    │
│  │   │   │ • HuggingFace│ │ • Debate     │ │ • Validation │                 │     │    │
│  │   │   │ • Web Search │ │   (Bull/Bear)│ │ • Benchmarks │                 │     │    │
│  │   │   │ • PubMed     │ │              │ │              │                 │     │    │
│  │   │   └──────────────┘ └──────────────┘ └──────────────┘                 │     │    │
│  │   │                                                                        │     │    │
│  │   └────────────────────────────────────────────────────────────────────────┘     │    │
│  │                                          │                                        │    │
│  │                                          ▼                                        │    │
│  │   ┌───────────────────────────────────────────────────────────────────────┐     │    │
│  │   │                      MCP TOOLS LAYER (Native)                          │     │    │
│  │   │                                                                        │     │    │
│  │   │   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │     │    │
│  │   │   │ ArXiv MCP   │ │ GitHub MCP  │ │ HuggingFace │ │ AWS Docs    │    │     │    │
│  │   │   │ Server      │ │ Server      │ │ MCP Server  │ │ MCP Server  │    │     │    │
│  │   │   └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘    │     │    │
│  │   │   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │     │    │
│  │   │   │ Tavily      │ │ Exa Search  │ │ Bright Data │ │ Custom MCP  │    │     │    │
│  │   │   │ Search/Crawl│ │             │ │ Scraper     │ │ Servers     │    │     │    │
│  │   │   └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘    │     │    │
│  │   │                                                                        │     │    │
│  │   └────────────────────────────────────────────────────────────────────────┘     │    │
│  │                                                                                  │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                          │                                               │
│                                          ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                         AGENTCORE MANAGED SERVICES                               │    │
│  │                                                                                  │    │
│  │   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐                   │    │
│  │   │  AgentCore      │ │  AgentCore      │ │  Code           │                   │    │
│  │   │  Memory         │ │  Gateway        │ │  Interpreter    │                   │    │
│  │   │                 │ │                 │ │                 │                   │    │
│  │   │  • User Prefs   │ │  • Lambda Tools │ │  • Python       │                   │    │
│  │   │  • Semantic     │ │  • Auth/AuthZ   │ │  • JavaScript   │                   │    │
│  │   │    Facts        │ │  • Rate Limits  │ │  • TypeScript   │                   │    │
│  │   │  • Session      │ │                 │ │  • Isolated     │                   │    │
│  │   │    Summaries    │ │                 │ │    Sandbox      │                   │    │
│  │   └─────────────────┘ └─────────────────┘ └─────────────────┘                   │    │
│  │                                                                                  │    │
│  │   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐                   │    │
│  │   │  Bedrock        │ │  Bedrock        │ │  CloudWatch     │                   │    │
│  │   │  Foundation     │ │  Knowledge      │ │  + X-Ray        │                   │    │
│  │   │  Models         │ │  Bases          │ │                 │                   │    │
│  │   │                 │ │                 │ │  • Metrics      │                   │    │
│  │   │  • Claude       │ │  • RAG          │ │  • Traces       │                   │    │
│  │   │  • Nova         │ │  • Embeddings   │ │  • Logs         │                   │    │
│  │   │  • Titan        │ │  • Vector Store │ │  • Alarms       │                   │    │
│  │   └─────────────────┘ └─────────────────┘ └─────────────────┘                   │    │
│  │                                                                                  │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                          │                                               │
│                                          ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                         AWS INFRASTRUCTURE SERVICES                              │    │
│  │                                                                                  │    │
│  │   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │    │
│  │   │ S3          │ │ DynamoDB    │ │ OpenSearch  │ │ EventBridge │              │    │
│  │   │ (Documents, │ │ (Metadata,  │ │ (Full-text  │ │ (Proactive  │              │    │
│  │   │  Artifacts) │ │  Sessions)  │ │  Search)    │ │  Triggers)  │              │    │
│  │   └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘              │    │
│  │                                                                                  │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Code Examples

### 1. Project Setup with FAST Template

```bash
# Clone the fullstack AgentCore template
git clone https://github.com/awslabs/fullstack-solution-template-for-agentcore
cd fullstack-solution-template-for-agentcore

# Install dependencies
cd infra-cdk && npm install && cd ..

# Deploy infrastructure (Cognito, API Gateway, Frontend)
cd infra-cdk && cdk bootstrap && cdk deploy && cd ..

# Deploy frontend
python scripts/deploy-frontend.py

# Your DOVA foundation is now running!
```

### 2. Basic Research Agent with MCP

```python
# dova/agents/research_agent.py
from strands import Agent
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters

class DOVAResearchAgent:
    """Multi-source research agent using MCP servers."""

    def __init__(self):
        # Initialize MCP clients for each data source
        self.arxiv_mcp = MCPClient(
            lambda: stdio_client(StdioServerParameters(
                command="uvx",
                args=["mcp-server-arxiv"]
            ))
        )

        self.github_mcp = MCPClient(
            lambda: stdio_client(StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-github"]
            ))
        )

        self.hf_mcp = MCPClient(
            lambda: stdio_client(StdioServerParameters(
                command="uvx",
                args=["mcp-server-huggingface"]
            ))
        )

    def create_agent(self) -> Agent:
        """Create the research agent with all MCP tools."""
        # Gather tools from all MCP servers
        all_tools = []

        with self.arxiv_mcp:
            all_tools.extend(self.arxiv_mcp.list_tools_sync())

        with self.github_mcp:
            all_tools.extend(self.github_mcp.list_tools_sync())

        with self.hf_mcp:
            all_tools.extend(self.hf_mcp.list_tools_sync())

        return Agent(
            system_prompt="""You are DOVA, a Deep Orchestrated Versatile Agent
            for research automation. Your capabilities include:

            1. Search ArXiv for academic papers and preprints
            2. Search GitHub for code implementations and repositories
            3. Search HuggingFace for models, datasets, and spaces

            When answering research queries:
            - Search multiple sources in parallel when possible
            - Cross-reference papers with their implementations
            - Provide citations and links to sources
            - Synthesize findings into coherent summaries
            """,
            tools=all_tools
        )

    async def research(self, query: str) -> str:
        """Execute a research query."""
        agent = self.create_agent()
        result = agent(query)
        return result.message

# Usage
if __name__ == "__main__":
    researcher = DOVAResearchAgent()
    result = researcher.research(
        "Find the latest papers on multi-agent LLM systems "
        "and their open-source implementations"
    )
    print(result)
```

### 3. Custom Source Management API

```python
# dova/services/sources.py
"""User-defined custom sources with quality learning."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class SourceType(Enum):
    BUILTIN = "builtin"      # arxiv, github, huggingface
    WEB_URL = "web_url"      # scrape web pages
    RSS_FEED = "rss_feed"    # RSS/Atom feeds
    API = "api"              # custom API endpoints

@dataclass
class QualityMetrics:
    """Implicit quality signals for a source."""
    query_count: int = 0
    click_count: int = 0
    save_count: int = 0
    avg_position_clicked: float = 0.0

    @property
    def quality_score(self) -> float:
        """Calculate quality score from implicit signals (0-1)."""
        if self.query_count == 0:
            return 0.5  # neutral for new sources
        click_rate = self.click_count / max(self.query_count, 1)
        save_rate = self.save_count / max(self.click_count, 1)
        position_score = 1.0 / (1 + self.avg_position_clicked / 10)
        return min(1.0, (click_rate * 0.4) + (save_rate * 0.3) + (position_score * 0.3))

@dataclass
class Source:
    """A research source (built-in or custom)."""
    id: str
    user_id: str
    name: str
    source_type: SourceType
    enabled: bool = True
    url: str = ""
    auth_type: str | None = None  # "bearer", "api_key"
    quality: QualityMetrics = field(default_factory=QualityMetrics)

class SourceRegistry:
    """Manages sources per user with quality tracking."""

    async def get_sources(self, user_id: str, enabled_only: bool = True) -> list[Source]:
        """Get all sources for a user, sorted by quality score."""
        # Combines built-in sources + user's custom sources
        # Sorts by quality_score (highest first)
        pass

    async def add_source(self, user_id: str, name: str,
                         source_type: SourceType, url: str) -> Source:
        """Add a custom source for a user."""
        pass

    async def record_interaction(self, user_id: str, source_id: str,
                                 interaction_type: str,  # "query", "click", "save"
                                 result_position: int | None = None) -> None:
        """Record an implicit quality signal."""
        pass

# API Usage Examples:
#
# GET  /api/v1/sources           - List all sources (built-in + custom)
# POST /api/v1/sources           - Add custom source
#      {"name": "HN", "source_type": "rss_feed",
#       "config": {"url": "https://news.ycombinator.com/rss"}}
# PUT  /api/v1/sources/{id}      - Update source (enable/disable)
# DELETE /api/v1/sources/{id}    - Delete custom source
# POST /api/v1/sources/interact  - Record interaction for quality learning
#      {"source_id": "custom_abc", "interaction_type": "click", "result_position": 2}
```

### 3. User Profiling with AgentCore Memory

```python
# dova/agents/profiling_agent.py
from strands import Agent
from strands_tools.agent_core_memory import AgentCoreMemoryToolProvider
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class UserProfile:
    user_id: str
    session_id: str
    namespace: str = "dova_profiles"

class DOVAProfileAgent:
    """User profiling agent using AgentCore Memory."""

    def __init__(
        self,
        memory_id: str,
        region: str = "us-west-2"
    ):
        self.memory_id = memory_id
        self.region = region

    def create_agent(self, profile: UserProfile) -> Agent:
        """Create a profiling agent for a specific user."""

        # Initialize AgentCore Memory provider
        memory_provider = AgentCoreMemoryToolProvider(
            memory_id=self.memory_id,
            actor_id=profile.user_id,
            session_id=profile.session_id,
            namespace=profile.namespace,
            region=self.region
        )

        return Agent(
            system_prompt="""You are the DOVA User Profiling Agent.
            Your responsibilities:

            1. RECORD user preferences, interests, and expertise levels
            2. RETRIEVE relevant user context for personalization
            3. UPDATE user profiles based on interactions
            4. MAINTAIN temporal awareness (recent vs long-term interests)

            Memory Actions:
            - action="record": Store new information about the user
            - action="retrieve": Search for relevant user context
            - action="list": List all stored memories
            - action="get": Get a specific memory by ID

            Always be privacy-conscious and only store relevant research preferences.
            """,
            tools=memory_provider.tools
        )

    async def record_preference(
        self,
        profile: UserProfile,
        preference: str
    ) -> str:
        """Record a user preference."""
        agent = self.create_agent(profile)
        result = agent(f"Record this user preference: {preference}")
        return result.message

    async def get_user_context(
        self,
        profile: UserProfile,
        query: str
    ) -> str:
        """Retrieve relevant user context for a query."""
        agent = self.create_agent(profile)
        result = agent(
            f"Retrieve any relevant user preferences or history for: {query}"
        )
        return result.message

# Usage
if __name__ == "__main__":
    profiler = DOVAProfileAgent(memory_id="dova-memory-123")

    profile = UserProfile(
        user_id="user-456",
        session_id="session-789"
    )

    # Record a preference
    profiler.record_preference(
        profile,
        "User is interested in transformer architectures and NLP"
    )

    # Get context for personalization
    context = profiler.get_user_context(
        profile,
        "multi-agent systems"
    )
```

### 4. Master Orchestrator with Multi-Agent Coordination

```python
# dova/agents/orchestrator.py
from strands import Agent, tool
from strands_tools import batch, agent_graph
from typing import List, Dict, Any
import asyncio

class DOVAOrchestrator:
    """Master orchestrator using Strands multi-agent patterns."""

    def __init__(
        self,
        research_agent: Agent,
        profiling_agent: Agent,
        validation_agent: Agent
    ):
        self.research_agent = research_agent
        self.profiling_agent = profiling_agent
        self.validation_agent = validation_agent

        # Create tool wrappers for sub-agents
        self._setup_agent_tools()

    def _setup_agent_tools(self):
        """Wrap sub-agents as tools for the orchestrator."""

        @tool
        def search_research(query: str) -> str:
            """Search for research papers, code, and models across multiple sources."""
            return self.research_agent(query).message

        @tool
        def get_user_context(user_id: str, query: str) -> str:
            """Retrieve user profile and preferences for personalization."""
            return self.profiling_agent(
                f"Get context for user {user_id} regarding: {query}"
            ).message

        @tool
        def validate_code(code: str, description: str) -> str:
            """Validate code in a secure sandbox environment."""
            return self.validation_agent(
                f"Validate this code:\n```\n{code}\n```\nDescription: {description}"
            ).message

        @tool
        def synthesize_findings(findings: List[str]) -> str:
            """Synthesize multiple research findings into a coherent summary."""
            combined = "\n\n".join(findings)
            return self.research_agent(
                f"Synthesize these findings into a coherent summary:\n{combined}"
            ).message

        self.tools = [
            search_research,
            get_user_context,
            validate_code,
            synthesize_findings,
            batch  # For parallel execution
        ]

    def create_orchestrator(self) -> Agent:
        """Create the master orchestrator agent."""
        return Agent(
            system_prompt="""You are the DOVA Master Orchestrator.

            Your role is to:
            1. UNDERSTAND user intent and decompose complex queries
            2. COORDINATE multiple specialized agents
            3. PERSONALIZE responses based on user context
            4. SYNTHESIZE findings from multiple sources
            5. VALIDATE code and implementations when needed

            Workflow for research queries:
            1. First, get user context for personalization
            2. Search across research sources (papers, code, models)
            3. If code is found, optionally validate it
            4. Synthesize all findings into a personalized response

            Use the batch tool to execute independent tasks in parallel.

            Always cite sources and provide actionable insights.
            """,
            tools=self.tools
        )

    async def process_query(
        self,
        user_id: str,
        query: str
    ) -> Dict[str, Any]:
        """Process a user query through the orchestration pipeline."""

        orchestrator = self.create_orchestrator()

        # Execute the orchestrated query
        result = orchestrator(f"""
        User ID: {user_id}
        Query: {query}

        Please:
        1. Get user context first
        2. Search for relevant research
        3. Synthesize personalized findings
        """)

        return {
            "response": result.message,
            "metrics": {
                "tokens_used": result.metrics.get("total_tokens", 0),
                "tools_called": len(result.tool_calls) if result.tool_calls else 0
            }
        }

# Usage
if __name__ == "__main__":
    from dova.agents.research_agent import DOVAResearchAgent
    from dova.agents.profiling_agent import DOVAProfileAgent
    from dova.agents.validation_agent import DOVAValidationAgent

    # Initialize sub-agents
    research = DOVAResearchAgent().create_agent()
    profiling = DOVAProfileAgent(memory_id="dova-mem").create_agent(...)
    validation = DOVAValidationAgent().create_agent()

    # Create orchestrator
    orchestrator = DOVAOrchestrator(research, profiling, validation)

    # Process a query
    result = asyncio.run(orchestrator.process_query(
        user_id="user-123",
        query="What are the best approaches for building multi-agent RAG systems?"
    ))

    print(result["response"])
```

### 4.5 ThinkingOrchestrator (Deliberation-First) *(New in v1.5)*

The ThinkingOrchestrator provides a deliberation-first approach that reasons about user needs before deciding which tools to use:

```python
# dova/agents/thinking_orchestrator.py
from dova.agents.base import BaseAgent
from dova.agents.user_model import UserModel, ExpertiseLevel, ResponseDepth
from dova.agents.conversation_context import ConversationContext

class ThinkingOrchestrator(BaseAgent):
    """Deliberation-first orchestrator that thinks before acting."""

    async def execute(self, task: AgentTask) -> AgentResult:
        # 1. Load user model and conversation context
        user_model = await self._load_user_model(task.user_id)
        context = await self._load_conversation_context(task.session_id)

        # 2. DELIBERATE - the key innovation
        deliberation = await self._deliberate(query, user_model, context)

        # 3. Execute based on deliberation decision
        if deliberation.action == ActionDecision.RESPOND_DIRECTLY:
            response = await self._respond_from_context(query, deliberation, user_model)
        elif deliberation.action == ActionDecision.USE_TOOLS:
            tool_results = await self._execute_selected_tools(deliberation)
            response = await self._synthesize_with_results(query, tool_results, user_model)
        else:  # CLARIFY
            response = deliberation.clarification_needed

        # 4. Update context and return
        await self._update_context(context, query, response)
        return self._wrap_result(task, True, data={"response": response})
```

**Deliberation Prompt:**
```python
DELIBERATION_PROMPT = """You are deciding how to help a user. Think carefully before acting.

USER QUERY: {query}

ABOUT THIS USER:
- Expertise: {expertise_areas}
- Preferred depth: {preferred_depth}
- Session goals: {session_goals}

CONVERSATION CONTEXT:
- Topic: {current_topic}
- Already discussed: {entities_discussed}

AVAILABLE TOOLS (use ONLY if needed):
- arxiv: Academic papers
- github: Code repositories
- huggingface: ML models/datasets
- web: Web search

THINK THROUGH:
1. What does the user ACTUALLY need?
2. Can I answer from existing context/knowledge?
3. If tools needed, which specific ones and why?

Respond with JSON:
{{
  "understanding": "what user actually needs",
  "can_answer_from_context": true/false,
  "tools_to_use": [{{"tool": "arxiv|github|huggingface|web", "rationale": "why"}}],
  "action": "respond_directly|use_tools|clarify"
}}"""
```

**User Model:**
```python
@dataclass
class UserModel:
    user_id: str
    expertise_areas: dict[str, ExpertiseLevel]  # {"transformers": EXPERT}
    preferred_depth: ResponseDepth  # BRIEF, STANDARD, DETAILED
    prefers_code_examples: bool
    formality: str  # "technical", "casual"
    current_goals: list[str]

class ExpertiseLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"
```

**Usage:**
```python
# CLI
dova interact --orchestrator thinking
dova research "explain attention" --orchestrator thinking

# In interactive mode
/orchestrator thinking
```

### 5. Code Validation with AgentCore Code Interpreter

```python
# dova/agents/validation_agent.py
from strands import Agent
from strands_tools.code_interpreter import AgentCoreCodeInterpreter

class DOVAValidationAgent:
    """Code validation agent using AgentCore Code Interpreter."""

    def __init__(self, region: str = "us-west-2"):
        self.code_interpreter = AgentCoreCodeInterpreter(region=region)

    def create_agent(self) -> Agent:
        """Create the validation agent."""
        return Agent(
            system_prompt="""You are the DOVA Validation Agent.

            Your responsibilities:
            1. VALIDATE code snippets from research papers
            2. TEST implementations for correctness
            3. BENCHMARK performance when relevant
            4. REPORT issues and suggestions

            When validating code:
            - First create a session for the validation task
            - Execute the code in the sandbox
            - Capture outputs, errors, and metrics
            - Provide clear pass/fail assessment with reasoning

            Always prioritize security - never execute obviously malicious code.
            """,
            tools=[self.code_interpreter.code_interpreter]
        )

    async def validate_snippet(
        self,
        code: str,
        language: str = "python",
        description: str = ""
    ) -> dict:
        """Validate a code snippet."""
        agent = self.create_agent()

        result = agent(f"""
        Please validate this {language} code:

        Description: {description}

        ```{language}
        {code}
        ```

        Steps:
        1. Create a validation session
        2. Execute the code
        3. Report results (success/failure, output, any errors)
        """)

        return {
            "validation_result": result.message,
            "success": "error" not in result.message.lower()
        }

# Usage
if __name__ == "__main__":
    validator = DOVAValidationAgent()

    result = validator.validate_snippet(
        code="""
import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers=6
        )

    def forward(self, x):
        return self.transformer(x)

# Test
model = SimpleTransformer()
x = torch.randn(10, 32, 512)
output = model(x)
print(f"Output shape: {output.shape}")
        """,
        language="python",
        description="Simple transformer encoder from research paper"
    )

    print(result)
```

### 6. Debate Agents (Bull vs Bear Pattern)

```python
# dova/agents/debate_agents.py
from strands import Agent, tool
from strands_tools import swarm
from typing import Tuple

class DOVADebateAgents:
    """Bull vs Bear debate pattern for balanced analysis."""

    def __init__(self, model_id: str = "anthropic.claude-sonnet-4-20250514-v1:0"):
        self.model_id = model_id

    def create_advocate_agent(self) -> Agent:
        """Create the Bull (advocate) agent."""
        return Agent(
            system_prompt="""You are the DOVA Advocate Agent (Bull).

            Your role is to:
            - Present the STRONGEST CASE for the proposed solution
            - Highlight benefits, advantages, and opportunities
            - Address potential concerns with counter-arguments
            - Be persuasive but factual

            Focus on: feasibility, innovation, competitive advantages,
            and positive outcomes.
            """
        )

    def create_critic_agent(self) -> Agent:
        """Create the Bear (critic) agent."""
        return Agent(
            system_prompt="""You are the DOVA Critic Agent (Bear).

            Your role is to:
            - Present LEGITIMATE CONCERNS about the proposed solution
            - Identify risks, challenges, and potential failures
            - Question assumptions and highlight gaps
            - Be critical but constructive

            Focus on: risks, costs, complexity, edge cases,
            and potential negative outcomes.
            """
        )

    def create_synthesis_agent(self) -> Agent:
        """Create the synthesis agent for balanced conclusions."""
        return Agent(
            system_prompt="""You are the DOVA Synthesis Agent.

            Your role is to:
            - Analyze arguments from both Advocate and Critic
            - Identify valid points from each perspective
            - Produce a BALANCED, NUANCED conclusion
            - Provide actionable recommendations

            Your output should help decision-makers understand:
            - Key benefits with confidence levels
            - Key risks with mitigation strategies
            - Overall recommendation with caveats
            """
        )

    async def debate(
        self,
        topic: str,
        context: str,
        rounds: int = 2
    ) -> dict:
        """Run a structured debate on a topic."""

        advocate = self.create_advocate_agent()
        critic = self.create_critic_agent()
        synthesis = self.create_synthesis_agent()

        debate_history = []

        # Initial positions
        advocate_position = advocate(
            f"Topic: {topic}\nContext: {context}\n\n"
            "Present your initial case FOR this approach."
        ).message
        debate_history.append(("Advocate", advocate_position))

        critic_position = critic(
            f"Topic: {topic}\nContext: {context}\n\n"
            f"The advocate argues:\n{advocate_position}\n\n"
            "Present your concerns and counter-arguments."
        ).message
        debate_history.append(("Critic", critic_position))

        # Additional rounds
        for round_num in range(rounds - 1):
            # Advocate responds to criticism
            advocate_response = advocate(
                f"The critic argues:\n{critic_position}\n\n"
                "Address these concerns and strengthen your case."
            ).message
            debate_history.append(("Advocate", advocate_response))

            # Critic responds
            critic_response = critic(
                f"The advocate responds:\n{advocate_response}\n\n"
                "Provide your final concerns and analysis."
            ).message
            debate_history.append(("Critic", critic_response))

            advocate_position = advocate_response
            critic_position = critic_response

        # Synthesis
        debate_summary = "\n\n".join([
            f"**{speaker}**: {content}"
            for speaker, content in debate_history
        ])

        final_synthesis = synthesis(
            f"Topic: {topic}\n\n"
            f"Debate Summary:\n{debate_summary}\n\n"
            "Provide a balanced synthesis with recommendations."
        ).message

        return {
            "topic": topic,
            "debate_history": debate_history,
            "synthesis": final_synthesis
        }

# Usage
if __name__ == "__main__":
    debate = DOVADebateAgents()

    result = asyncio.run(debate.debate(
        topic="Using ColPali for multi-modal RAG in enterprise documents",
        context="""
        We're evaluating ColPali as a replacement for traditional
        chunking + embedding approaches for RAG on enterprise documents
        containing tables, charts, and mixed layouts.
        """,
        rounds=2
    ))

    print("=== SYNTHESIS ===")
    print(result["synthesis"])
```

### 7. Proactive Recommendations with EventBridge

```python
# dova/agents/proactive_agent.py
import boto3
import json
from strands import Agent
from strands_tools import tavily_search
from strands_tools.agent_core_memory import AgentCoreMemoryToolProvider

class DOVAProactiveAgent:
    """Proactive recommendation agent triggered by EventBridge."""

    def __init__(
        self,
        memory_id: str,
        region: str = "us-west-2"
    ):
        self.memory_id = memory_id
        self.region = region
        self.eventbridge = boto3.client('events', region_name=region)

    def create_monitoring_rule(self, rule_name: str, schedule: str):
        """Create an EventBridge rule for periodic monitoring."""

        self.eventbridge.put_rule(
            Name=rule_name,
            ScheduleExpression=schedule,  # e.g., "rate(6 hours)"
            State='ENABLED',
            Description='DOVA proactive monitoring for new research'
        )

    def create_agent(self, user_id: str, session_id: str) -> Agent:
        """Create the proactive recommendation agent."""

        memory_provider = AgentCoreMemoryToolProvider(
            memory_id=self.memory_id,
            actor_id=user_id,
            session_id=session_id,
            namespace="proactive_recommendations",
            region=self.region
        )

        return Agent(
            system_prompt="""You are the DOVA Proactive Agent.

            Your responsibilities:
            1. MONITOR for new research matching user interests
            2. EVALUATE relevance against user profile
            3. GENERATE personalized alerts when high-relevance content found
            4. RECORD delivered recommendations to avoid duplicates

            Relevance scoring:
            - 0.9+: Immediate notification
            - 0.7-0.9: Include in daily digest
            - <0.7: Skip

            Always explain WHY content is relevant to the user.
            """,
            tools=[
                tavily_search,
                *memory_provider.tools
            ]
        )

    async def check_for_updates(
        self,
        user_id: str,
        interests: list[str]
    ) -> list[dict]:
        """Check for new content matching user interests."""

        agent = self.create_agent(user_id, f"proactive-{user_id}")

        recommendations = []

        for interest in interests:
            result = agent(f"""
            Search for the latest research on: {interest}

            Then:
            1. Check user memory for previously recommended items
            2. Filter out duplicates
            3. Score relevance (0-1) for new items
            4. Return only high-relevance items (>0.7)
            5. Record any recommendations you make
            """)

            recommendations.append({
                "interest": interest,
                "findings": result.message
            })

        return recommendations

# Lambda handler for EventBridge trigger
def lambda_handler(event, context):
    """AWS Lambda handler for proactive monitoring."""

    agent = DOVAProactiveAgent(
        memory_id=os.environ["DOVA_MEMORY_ID"]
    )

    # Get users to check (from DynamoDB or event payload)
    users = event.get("users", [])

    results = []
    for user in users:
        recommendations = asyncio.run(agent.check_for_updates(
            user_id=user["user_id"],
            interests=user["interests"]
        ))

        if recommendations:
            # Send notifications (SNS, email, etc.)
            send_notifications(user["user_id"], recommendations)
            results.append({
                "user_id": user["user_id"],
                "recommendations_sent": len(recommendations)
            })

    return {"statusCode": 200, "body": json.dumps(results)}
```

### 8. Full DOVA Application Entry Point

```python
# dova/main.py
from strands import Agent
from strands.models import BedrockModel
from dova.agents.orchestrator import DOVAOrchestrator
from dova.agents.research_agent import DOVAResearchAgent
from dova.agents.profiling_agent import DOVAProfileAgent, UserProfile
from dova.agents.validation_agent import DOVAValidationAgent
from dova.agents.debate_agents import DOVADebateAgents
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="DOVA - Deep Orchestrated Versatile Agent Platform")

# Configuration
MEMORY_ID = "dova-memory-production"
REGION = "us-west-2"

# Initialize agents
research_agent = DOVAResearchAgent()
profiling_agent = DOVAProfileAgent(memory_id=MEMORY_ID, region=REGION)
validation_agent = DOVAValidationAgent(region=REGION)
debate_agents = DOVADebateAgents()

class ResearchQuery(BaseModel):
    query: str
    user_id: str
    session_id: str
    include_validation: bool = False
    include_debate: bool = False

class ResearchResponse(BaseModel):
    response: str
    sources: list[str] = []
    validation_result: str | None = None
    debate_synthesis: str | None = None

@app.post("/research", response_model=ResearchResponse)
async def research(query: ResearchQuery):
    """Execute a deep research query."""

    # Create user profile
    profile = UserProfile(
        user_id=query.user_id,
        session_id=query.session_id
    )

    # Get user context
    user_context = await profiling_agent.get_user_context(
        profile, query.query
    )

    # Create orchestrator with user context
    orchestrator = DOVAOrchestrator(
        research_agent=research_agent.create_agent(),
        profiling_agent=profiling_agent.create_agent(profile),
        validation_agent=validation_agent.create_agent()
    )

    # Execute research
    result = await orchestrator.process_query(
        user_id=query.user_id,
        query=f"Context: {user_context}\n\nQuery: {query.query}"
    )

    response = ResearchResponse(response=result["response"])

    # Optional: Validate code findings
    if query.include_validation:
        # Extract code from response and validate
        # (simplified for example)
        validation = await validation_agent.validate_snippet(
            code="# extracted code",
            description="Code from research"
        )
        response.validation_result = validation["validation_result"]

    # Optional: Run debate analysis
    if query.include_debate:
        debate_result = await debate_agents.debate(
            topic=query.query,
            context=result["response"],
            rounds=2
        )
        response.debate_synthesis = debate_result["synthesis"]

    # Record interaction for future personalization
    await profiling_agent.record_preference(
        profile,
        f"User researched: {query.query}"
    )

    return response

@app.get("/health")
async def health():
    return {"status": "healthy", "platform": "DOVA"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Deployment with AgentCore

### Deploy Agent to AgentCore Runtime

```bash
# Install AgentCore toolkit
pip install bedrock-agentcore-starter-toolkit

# Configure your agent
agentcore configure -e dova/main.py

# Deploy to AgentCore Runtime (serverless)
agentcore launch

# Test the deployed agent
agentcore invoke '{"query": "Latest advances in multi-agent systems", "user_id": "test-user"}'

# View logs
agentcore logs

# Destroy when done
agentcore destroy
```

### Infrastructure as Code (CDK)

```typescript
// infra/lib/dova-stack.ts
import * as cdk from 'aws-cdk-lib';
import * as cognito from 'aws-cdk-lib/aws-cognito';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as s3 from 'aws-cdk-lib/aws-s3';

export class DOVAStack extends cdk.Stack {
  constructor(scope: cdk.App, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // User Pool for authentication
    const userPool = new cognito.UserPool(this, 'DOVAUserPool', {
      selfSignUpEnabled: true,
      signInAliases: { email: true },
      mfa: cognito.Mfa.OPTIONAL,
    });

    // DynamoDB for session metadata
    const sessionsTable = new dynamodb.Table(this, 'DOVASessions', {
      partitionKey: { name: 'user_id', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'session_id', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
    });

    // S3 for document storage
    const documentsBucket = new s3.Bucket(this, 'DOVADocuments', {
      encryption: s3.BucketEncryption.S3_MANAGED,
      versioned: true,
    });

    // Outputs
    new cdk.CfnOutput(this, 'UserPoolId', { value: userPool.userPoolId });
    new cdk.CfnOutput(this, 'SessionsTableName', { value: sessionsTable.tableName });
    new cdk.CfnOutput(this, 'DocumentsBucketName', { value: documentsBucket.bucketName });
  }
}
```

---

## Cost Comparison

### Original DOVA Design vs Strands + AgentCore

| Component | Original Design | Strands + AgentCore | Savings |
|-----------|-----------------|---------------------|---------|
| **Compute** | K8s cluster ($15-25K) | AgentCore Runtime ($3-8K) | 60-70% |
| **Orchestration** | Temporal + Airflow ($2-4K) | AgentCore Runtime (included) | 100% |
| **Memory/Profile** | PostgreSQL + Redis ($2-4K) | AgentCore Memory ($0.5-2K) | 50-75% |
| **Sandbox** | Custom K8s pods ($3-8K) | Code Interpreter ($1-3K) | 60% |
| **Vector DB** | Qdrant/Pinecone ($2-5K) | Bedrock KB (included) | 80% |
| **LLM Costs** | $10-30K | $5-15K (Bedrock pricing) | 50% |
| **Ops Overhead** | 2-3 engineers | 0.5-1 engineer | 60-80% |

### Monthly Cost Estimate (Strands + AgentCore)

| Phase | Monthly Cost | Notes |
|-------|-------------|-------|
| **MVP (Month 1-3)** | $3,000 - $8,000 | AgentCore + basic usage |
| **Growth (Month 4-6)** | $8,000 - $18,000 | Increased users, memory |
| **Scale (Month 7+)** | $15,000 - $35,000 | Full feature set |

**Total Savings: 50-60% compared to original design**

---

## Implementation Roadmap (Revised)

### Phase 0: Setup (1 week)
- [ ] Clone FAST template
- [ ] Deploy base infrastructure
- [ ] Verify Cognito + frontend working

### Phase 1: Core Agents (3 weeks)
- [ ] Implement research agent with MCP
- [ ] Implement profiling agent with AgentCore Memory
- [ ] Basic orchestrator

### Phase 2: Multi-Source Research (3 weeks)
- [ ] Add ArXiv, GitHub, HuggingFace MCP servers
- [ ] Implement parallel search with batch tool
- [ ] Result synthesis

### Phase 3: Validation & Sandbox (2 weeks)
- [ ] Integrate Code Interpreter
- [ ] Implement validation agent
- [ ] Code testing workflows

### Phase 4: Advanced Features (3 weeks)
- [ ] Debate agents (Bull/Bear)
- [ ] Proactive recommendations
- [ ] Advanced personalization

### Phase 5: Production (2 weeks)
- [ ] Load testing
- [ ] Security review
- [ ] Documentation
- [ ] Production deployment

**Total: ~14 weeks (vs 40+ weeks original estimate)**

---

## Key Resources

### Official Documentation
- [Strands Agents Documentation](https://strandsagents.com)
- [Amazon Bedrock AgentCore Guide](https://docs.aws.amazon.com/bedrock-agentcore)
- [FAST Template Repository](https://github.com/awslabs/fullstack-solution-template-for-agentcore)

### Sample Repositories
- [Strands SDK Python](https://github.com/strands-agents/sdk-python) - 5K+ ⭐
- [Strands Tools](https://github.com/strands-agents/tools) - 900+ ⭐
- [Strands Samples](https://github.com/strands-agents/samples) - 600+ ⭐
- [AgentCore Multi-Framework Examples](https://github.com/danilop/agentcore-multi-framework-examples)

### Training Resources
- [Getting Started with Strands Agents Course](https://github.com/aws-samples/sample-getting-started-with-strands-agents-course)
- [AWS Blog: Introducing Strands Agents](https://aws.amazon.com/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/)

---

## Conclusion

Implementing DOVA with **Strands Agents SDK + Amazon Bedrock AgentCore** provides:

| Benefit | Impact |
|---------|--------|
| **Faster Development** | 60%+ reduction in custom code |
| **Lower Costs** | 50%+ reduction in infrastructure |
| **Production Ready** | Enterprise security out of the box |
| **Maintainability** | Managed services, less ops burden |
| **Extensibility** | Native MCP support for future integrations |
| **Community** | Active open-source ecosystem |

**Recommended Next Steps:**
1. Clone the FAST template and deploy baseline
2. Implement the research agent with MCP servers
3. Add AgentCore Memory for user profiling
4. Iterate based on user feedback
