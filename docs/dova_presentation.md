# DOVA: Deliberation-First Multi-Agent Orchestration for Autonomous Research Automation

**Paper:** arXiv:2603.13327 — Aaron Shen, Alfred Shen (March 2026)

---

## Table of Contents

1. [Overview](#overview)
2. [Algorithm 1: ReAct Reasoning with Self-Reflection](#algorithm-1-react-reasoning-with-self-reflection)
3. [Algorithm 2: Deliberation-First Orchestration](#algorithm-2-deliberation-first-orchestration)
4. [Algorithm 3: Hybrid Collaborative Reasoning](#algorithm-3-hybrid-collaborative-reasoning)
5. [Algorithm 4: Adaptive Multi-Tiered Thinking](#algorithm-4-adaptive-multi-tiered-thinking)
6. [Algorithm 5: Query Intent Classification & Source Routing](#algorithm-5-query-intent-classification--source-routing)
7. [Algorithm 6: Multi-Round Adversarial Debate](#algorithm-6-multi-round-adversarial-debate)
8. [Algorithm 7: Ensemble Aggregation Methods](#algorithm-7-ensemble-aggregation-methods)
9. [Multi-Component Confidence Scoring](#multi-component-confidence-scoring)
10. [MMR-Enhanced Semantic Memory Search](#mmr-enhanced-semantic-memory-search)
11. [Global Optimization Objective](#global-optimization-objective)
12. [Ablation Study Results](#ablation-study-results)
13. [Reasoning Mode Ladder](#reasoning-mode-ladder)
14. [End-to-End Flow](#end-to-end-flow)

---

## Overview

DOVA implements **7 distinct reasoning algorithms** organized around three core innovations:

1. **Deliberation-First Orchestration** — explicit meta-reasoning precedes tool invocation, informed by a persistent user model and entity-aware conversation context. Unlike conventional agent frameworks that eagerly invoke tools, DOVA first *reasons about whether tools are needed at all*, eliminating unnecessary computation.

2. **Hybrid Collaborative Reasoning** — a composable three-phase pipeline unifying ensemble diversity, blackboard transparency, and iterative refinement. Prior multi-agent systems typically use one collaboration paradigm; DOVA chains all three to progressively distill high-confidence answers.

3. **Adaptive Multi-Tiered Thinking** — a six-level token-budget allocation scheme achieving 40-60% inference cost reduction on simple tasks while preserving deep reasoning for complex tasks. This treats reasoning depth as a *tunable resource*, not a fixed parameter.

### Architecture

Five specialized agents (Research, Profiling, Validation, Synthesis, Debate) share two common capabilities via mixins: a ReAct reasoning loop with self-reflection, and tiered semantic memory. A model tiering system routes tasks to appropriate model configurations by complexity.

---

## Algorithm 1: ReAct Reasoning with Self-Reflection

### Idea

Extend the standard ReAct (Reason + Act) loop with an explicit **self-reflection** step. Rather than stopping after the final observation, the agent critiques its own answer and refines it — catching errors that accumulate across reasoning steps.

### Formulation

Given problem $q$, max iterations $N$, and reflect flag $\varphi$:

$$\tau = \{(s_i, a_i, o_i)\}_{i=1}^{n}, \quad n \leq N$$

Each iteration:

$$
(s_i, a_i, c_i) = \text{Think}(q, \tau_{<i}, \text{pad})
$$

If $a_i = \text{conclude}$, terminate. Otherwise execute $o_i = \text{Act}(a_i)$ and append to the scratchpad.

**Post-loop reflection** (when $\varphi = \text{true}$):

$$
(r', \text{critique}) = \text{Reflect}(r, q, \tau)
$$

**Trace confidence** is the mean over per-step confidences:

$$
\bar{c}(\tau) = \frac{1}{|\{c_i\}|} \sum_i c_i, \quad c_i \in [0, 1]
$$

### Novelty

- The scratchpad serves as **working memory** across iterations, preventing the agent from losing context across long reasoning chains
- Self-reflection acts as a lightweight self-consistency check without requiring multiple independent samples
- The confidence trace provides an intrinsic quality signal that downstream components (ensemble, quality gate) can use

### Benefit

Without ReAct, confidence collapses from 0.82 to 0.58 and source coverage drops from 0.90 to 0.45 — the single largest degradation among all ablations.

---

## Algorithm 2: Deliberation-First Orchestration

### Idea

Before invoking any tool, perform **meta-reasoning** that considers the user model, conversation entities, and available tools to decide whether external tools are even necessary. This inverts the conventional pattern where agents default to tool use and instead makes tool invocation a *deliberate choice*.

### Formulation

Given query $q$, user model $u$, conversation context $\xi$, and available sources $\mathcal{D}'$:

$$
\delta = \text{LLM\_Deliberate}\bigl(q, \; \text{Expertise}(u), \; \text{Entities}(\xi), \; \text{RecentTurns}(\xi, k), \; \text{Tools}(\mathcal{D}')\bigr)
$$

The deliberation yields an action decision $\delta.\text{action} \in \{\text{RESPOND\_DIRECTLY}, \; \text{USE\_TOOLS}, \; \text{CLARIFY}\}$.

**Mandatory trigger override:** certain query patterns (temporal keywords, specificity markers, real-time data needs) force $\delta.\text{action} = \text{USE\_TOOLS}$ regardless of the LLM's recommendation, ensuring factual grounding when stakes are high.

**Cost reduction proposition:**

$$
\text{Savings} = f_d \cdot \bar{c}_{\text{tool}}
$$

where $f_d$ is the fraction of queries choosing RESPOND_DIRECTLY and $\bar{c}_{\text{tool}}$ is the average tool-call cost.

### Novelty

- **User-model-aware gating**: the deliberation considers user expertise level and communication preferences — an expert asking a follow-up on a discussed entity gets a direct answer; a novice asking the same question may receive tool-enriched context
- **Entity-aware context**: tracked conversation entities prevent redundant searches for topics already discussed
- **Mandatory triggers as safety rails**: hard-coded patterns override the LLM's judgment for queries where tool grounding is non-negotiable

### Benefit

Removing deliberation causes **19% latency increase** (unnecessary tool calls fire) and **27% token efficiency drop**. The deliberation layer acts as the system's primary cost gate.

---

## Algorithm 3: Hybrid Collaborative Reasoning

### Idea

Chain three complementary collaboration paradigms into a single pipeline: (1) **ensemble** for diversity, (2) **blackboard** for transparent evidence-based refinement, (3) **iterative refinement** for convergence. Each phase addresses a different failure mode of multi-agent reasoning.

### Formulation

Given problem $q$, agents $\{A_i\}_{i=1}^{n}$, and max iterations $K$:

**Phase 1 — Ensemble (Diversity):**

Each agent solves independently in parallel:

$$
(r_i, c_i) = A_i.\text{solve}(q, \xi), \quad \forall i
$$

Agreement score measures confidence variance:

$$
\mathcal{A}(c_1, \ldots, c_n) = \max\bigl(0, \; 1 - \text{Var}(c_1, \ldots, c_n)\bigr)
$$

**Phase 2 — Blackboard (Transparency):**

The ensemble answer is posted as a hypothesis; dissenting views become evidence. Each post $p$ has a weighted confidence:

$$
w(p) = c_{\text{base}}(p) \cdot \frac{1 + \bar{a}(p)}{2}
$$

where $\bar{a}(p)$ is the mean peer agreement vote, $v_{\text{agree}} \in [-1, 1]$. The blackboard is then synthesized into $r_{\text{bb}}$.

**Phase 3 — Iterative Refinement (Convergence):**

$$
r^* = \text{IterRefine}(r_{\text{bb}}, \; \{A_1, A_2\}, \; \min(2, K))
$$

Final confidence combines both phases:

$$
c^* = \frac{\bar{c}_{\text{ens}} + c_{\text{iter}}}{2}
$$

### Five Collaboration Modes

The pipeline is composable — each phase can run independently:

| Mode | Approach |
|------|----------|
| **Blackboard** | Hypothesize, gather evidence, synthesize |
| **Ensemble** | Parallel solve + synthesis aggregation |
| **Iterative** | Round-robin refinement across agents |
| **Hybrid** | Full 3-phase pipeline (Ensemble → Blackboard → Iterative) |
| **Tool-Augmented** | Tool planning + hybrid reasoning with enriched context |

### Novelty

- **Composability**: unlike prior work that commits to a single collaboration paradigm, DOVA chains them — ensemble catches blind spots, the blackboard makes evidence transparent, and iterative refinement drives convergence
- **Peer voting on the blackboard**: posts are weighted not just by the author's confidence but by peer agreement, surfacing consensus organically
- **Tool-augmented mode**: injects external data into the collaborative reasoning loop, grounding multi-agent debate in facts

### Benefit

Collaboration removal causes the **largest quality drop** of any component: $-0.14$ confidence, $-0.25$ coverage. The three-phase design is the primary driver of answer quality.

---

## Algorithm 4: Adaptive Multi-Tiered Thinking

### Idea

Treat the LLM's reasoning budget as a **tunable resource** rather than a fixed parameter. A classification question needs 1K tokens of reasoning; a complex research synthesis needs 65K. Allocating a fixed budget wastes tokens on simple tasks and starves complex ones.

### Formulation

The budget function composes three signals:

$$
B(t, h, q) = \text{Bud}\bigl[\text{clamp}\bigl(\beta(t) + \alpha(h) + \gamma(q), \; 0, \; 5\bigr)\bigr]
$$

where:
- $\beta(t)$: task type $\to$ base level index $\{0, \ldots, 5\}$
- $\alpha(h)$: complexity hint $\to$ adjustment $\{-1, 0, +1, +2\}$
- $\gamma(q)$: query length proxy $\to$ adjustment $\{-1, 0, +1\}$

### Six Thinking Levels

The budget array $\text{Bud}$ follows approximately $4\times$ geometric scaling:

| Level | Index | Token Budget | Typical Task |
|-------|-------|-------------|-------------|
| OFF | 0 | 0 | Embeddings |
| MINIMAL | 1 | 1,024 | Classification |
| LOW | 2 | 4,096 | Summarization |
| MEDIUM | 3 | 16,384 | Code generation |
| HIGH | 4 | 32,768 | Reasoning, research |
| XHIGH | 5 | 65,536 | Complex analysis |

### Adjustment Rules

| Signal | Condition | $\Delta$ |
|--------|-----------|----------|
| Complexity hint | simple | $-1$ |
| Complexity hint | complex | $+1$ |
| Complexity hint | very_complex | $+2$ |
| Query length | $|q| > 2000$ chars | $+1$ |
| Query length | $|q| < 50$ chars | $-1$ |

### Novelty

- **Two-dimensional cost optimization**: adaptive thinking depth (within a model) complements model tier routing (which model), optimizing along both axes simultaneously
- **Geometric scaling**: the $\sim4\times$ budget steps mean each level-up roughly quadruples reasoning capacity, matching the empirical observation that reasoning difficulty grows exponentially
- **Intentional overspend**: for research and reasoning tasks, the system *deliberately* allocates more tokens than a fixed-medium baseline — spending more where quality demands it

### Benefit

Under realistic workloads where 40-60% of queries are simple, aggregate token savings reach **40-60%**. Fixing the budget at Medium causes a **32% token efficiency loss** with only minor confidence improvement.

---

## Algorithm 5: Query Intent Classification & Source Routing

### Idea

Classify query intent via lightweight keyword scoring to route searches to the most relevant data sources. This avoids querying all sources for every query — a news question needs only web search, while a technical question benefits from arXiv, GitHub, and HuggingFace.

### Formulation

$$
t^*(q) = \arg\max_{t \in \mathcal{T}} \left[ \sum_{k \in \mathcal{K}_t} \mathbb{1}[k \in q_\downarrow] + \text{bonus}(q, t) \right]
$$

where $q_\downarrow$ is the lowercased query, $\mathcal{K}_t$ is the keyword set for type $t$, and:

$$
\text{bonus}(q, \text{biographical}) = 2 \cdot \mathbb{1}[\text{is\_person}(q)]
$$

Intent types $\mathcal{T} = \{\text{technical}, \; \text{news}, \; \text{biographical}, \; \text{factual}, \; \text{general}\}$.

### Source Routing Table

| Intent | ArXiv | GitHub | HuggingFace | Web |
|--------|:-----:|:------:|:-----------:|:---:|
| Technical | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| News | | | | $\checkmark$ |
| Biographical | | | | $\checkmark$ |
| Factual | $\checkmark$ | | | $\checkmark$ |
| General | $\checkmark$ | | $\checkmark$ | $\checkmark$ |

### Relevance Scoring

Results from each source are ranked by social proof signals:

$$
s_{\text{GitHub}}(r) = \frac{\text{stars}(r)}{1000}, \qquad s_{\text{HF}}(r) = \frac{\text{downloads}(r)}{100000}
$$

### Novelty

- **Zero-cost classification**: keyword matching requires no LLM call, keeping the routing overhead negligible
- **Person-indicator boost**: biographical queries get a $2\times$ score multiplier when a person entity is detected, preventing misclassification of "Who invented transformers?" as technical
- **Composable with deliberation**: intent classification provides a fast initial signal; the deliberation layer (Algorithm 2) can override or refine the source selection

### Benefit

Focused source routing eliminates irrelevant API calls and reduces latency. A technical query skips web-only sources; a news query avoids expensive arXiv/GitHub searches.

---

## Algorithm 6: Multi-Round Adversarial Debate

### Idea

For evaluative queries ("compare X vs Y", "should we use X?"), deploy two adversarial agents — a **Bull** (advocate) and a **Bear** (critic) — in sequential rounds. Each agent must engage with the opponent's prior arguments, forcing genuine dialectic rather than parallel monologues.

### Formulation

Given topic $q$, context $\xi$, and $R$ rounds (default $R = 2$):

$$
\text{For } r = 1, \ldots, R: \quad
\begin{cases}
b_r = \text{Bull}(q, \xi, \{k_1, \ldots, k_{r-1}\}) \\
k_r = \text{Bear}(q, \xi, \{b_1, \ldots, b_r\})
\end{cases}
$$

Final output:

$$
(\text{summary}, \; \text{strengths}, \; \text{concerns}, \; c) = \text{Synthesize}(\{b_r\}, \{k_r\})
$$

The synthesis identifies:
- **Surviving bull arguments** — not successfully rebutted
- **Validated bear concerns** — not adequately addressed
- **Overall confidence** — reflecting the balance of evidence

### Novelty

- **Sequential conditioning**: each agent sees *all* prior opponent arguments, not just the last round — this prevents circular argumentation and forces escalating specificity
- **Automatic trigger detection**: evaluative queries are detected via pattern matching ("compare", "vs", "tradeoffs", "pros and cons", "which is better"), seamlessly routing to debate mode without user intervention
- **Structured output**: rather than a free-form debate transcript, the synthesis produces actionable categories (strengths, concerns, recommendation)

### Benefit

Debate provides balanced analysis for decision-support queries that would otherwise receive one-sided answers from a single agent. The adversarial structure surfaces risks and limitations that a cooperative system might suppress.

---

## Algorithm 7: Ensemble Aggregation Methods

### Idea

Provide four strategies for combining multi-agent outputs, ranging from simple selection to LLM-powered synthesis. The choice of aggregation method adapts to the task: high-stakes decisions use synthesis; quick lookups use best-of.

### Four Strategies

**Best-of:** Select the highest-confidence answer. Other answers with $c > 0.3$ become dissenting views.

**Vote:** Weighted voting by confidence:

$$
\bar{c} = \frac{1}{n} \sum_{i=1}^{n} c_i
$$

**Union:** Concatenate all answers with their confidence labels, preserving full diversity.

**Synthesis (default):** An LLM synthesizes all perspectives — extracting strongest points, resolving contradictions, and flagging disagreements.

### Agreement Score (All Methods)

$$
\mathcal{A}(c_1, \ldots, c_n) = \max\bigl(0, \; 1 - \text{Var}(c_1, \ldots, c_n)\bigr)
$$

Range: $0$ (total disagreement) to $1$ (perfect agreement). Low variance in agent confidences signals consensus.

### Novelty

- **Pluggable aggregation**: the same ensemble infrastructure supports four strategies, selectable per-task
- **Dissenting view preservation**: even in best-of mode, minority opinions above a threshold are retained — preventing premature consensus
- **Agreement as a meta-signal**: the variance-based agreement score flows into downstream quality evaluation, triggering refinement when agents disagree

### Benefit

Synthesis aggregation is the default because it produces the highest-quality merged answers, but lighter methods (best-of, vote) are available when latency matters more than thoroughness.

---

## Multi-Component Confidence Scoring

### Idea

Evaluate answer quality through four orthogonal signals, producing a composite score that gates whether the system accepts or refines its answer. This replaces ad-hoc quality checks with a principled, weighted multi-factor assessment.

### Formulation

$$
C(r, p) = \frac{\sum_k w_k \cdot f_k(r, p)}{\sum_k w_k}
$$

| Component | Formula | Signal |
|-----------|---------|--------|
| Length | $f_{\text{len}}(r) = \text{clip}\!\left(\frac{|r|}{\tau_{\text{len}}}, \; 0.2, \; 1.0\right)$ | Penalizes too-short answers |
| Refusal | $f_{\text{ref}}(r) = 1 - 0.7 \cdot \mathbb{1}[\exists k \in \mathcal{K}_{\text{ref}} : k \subseteq r]$ | Detects refusal phrases |
| Format | $f_{\text{fmt}}(r, \varphi) = \text{format\_check}(r, \varphi)$ | Validates expected structure |
| Relevance | $f_{\text{rel}}(r, p) = \min\!\left(1, \; \frac{|\text{kw}(r) \cap \text{kw}(p)|}{0.3 \cdot |\text{kw}(p)|}\right)$ | Keyword overlap with query |

### Quality Gate Thresholds

| Condition | Action |
|-----------|--------|
| $C(r, p) \geq 0.7$ | Accept answer |
| $0.5 \leq C(r, p) < 0.7$ | Iterative query refinement (up to 2 rounds) |
| $C(r, p) < 0.5$ | Accept with quality warning |

### Novelty

The four components are deliberately orthogonal: length catches vacuous answers, refusal catches safety-filter false positives, format catches structural errors, and relevance catches off-topic drift. The iterative refinement loop at $C < 0.7$ gives the system a second chance before surfacing a low-quality answer.

---

## MMR-Enhanced Semantic Memory Search

### Idea

Use Maximal Marginal Relevance (MMR) for memory retrieval to balance **relevance** (finding the most pertinent memories) with **diversity** (avoiding redundant retrievals). Standard top-$k$ similarity search often returns near-duplicate memories; MMR ensures each retrieved item adds new information.

### Formulation

$$
\text{MMR}(d_i) = \lambda \cdot \text{sim}(d_i, q) - (1 - \lambda) \cdot \max_{d_j \in S} \text{sim}(d_i, d_j)
$$

where:
- $\text{sim}(a, b) = \frac{a \cdot b}{\|a\| \, \|b\|}$ (cosine similarity)
- $\lambda = 0.5$ balances relevance vs. diversity
- $S$ = already-selected result set

Greedy selection: at each step, pick $d^* = \arg\max_{d \in \text{candidates}} \text{MMR}(d)$, add to $S$, remove from candidates. Repeat $k$ times.

### Three Memory Tiers

| Tier | Persistence | Purpose |
|------|------------|---------|
| Short-term | 24 hours | Session context, recent interactions |
| Long-term | Persistent | User knowledge, preferences |
| Procedural | Persistent | Reusable skills and patterns |

### Novelty

Memory retrieval feeds into the deliberation layer (Algorithm 2), providing entity-aware context that helps the system decide whether tools are needed. The diversity guarantee from MMR ensures that retrieved memories span different topics rather than clustering around the most recent interaction.

---

## Global Optimization Objective

DOVA frames its overall goal as a constrained optimization:

$$
r^* = \arg\max_{r \in \mathcal{R}} \; C(r, q) \cdot \text{Cov}(r, \mathcal{D}) \quad \text{s.t.} \quad \text{cost}(r) \leq B(q)
$$

- $C(r, q)$: answer confidence (multi-component score)
- $\text{Cov}(r, \mathcal{D})$: source coverage across available data sources
- $B(q)$: adaptive thinking budget from Algorithm 4

This formulation makes explicit the three-way tradeoff between **quality**, **breadth**, and **cost**. The adaptive budget $B(q)$ is not a hard limit but a guide — the system intentionally overspends on complex tasks where quality demands it.

---

## Ablation Study Results

### Seven Configurations

| Configuration | Confidence $\uparrow$ | Coverage $\uparrow$ | Token Eff. $\uparrow$ | Latency (s) $\downarrow$ |
|--------------|:---:|:---:|:---:|:---:|
| **DOVA-Full** | **0.82** | **0.90** | **0.71** | **12.4** |
| $-$Collaboration | 0.68 | 0.65 | 0.74 | 6.1 |
| $-$Thinking (fixed Med) | 0.79 | 0.88 | 0.48 | 11.8 |
| $-$Memory | 0.75 | 0.85 | 0.65 | 11.2 |
| $-$Deliberation | 0.77 | 0.90 | 0.52 | 14.8 |
| $-$Self-Eval | 0.70 | 0.88 | 0.69 | 10.1 |
| $-$ReAct (single pass) | 0.58 | 0.45 | 0.80 | 3.2 |
| Single-LLM baseline | 0.52 | 0.00 | 0.85 | 1.8 |

### Key Findings

1. **Collaboration is the most impactful component**: $-0.14$ confidence, $-0.25$ coverage when removed
2. **Self-eval prevents unnecessary retries**: without it, refinement rate rises from 18% to 35%
3. **Adaptive thinking is a pure efficiency win**: 32% token savings with minimal confidence impact
4. **Deliberation is the cost gate**: removing it causes 19% more latency and 27% less efficiency
5. **ReAct is foundational**: single-pass causes confidence collapse ($0.82 \to 0.58$)

### Component Interaction Effects

- **Deliberation $\times$ Collaboration**: removing both is worse than the sum of individual removals — deliberation gatekeeps expensive collaboration, so without the gate, collaboration costs explode
- **Memory $\times$ Self-Eval**: memory context improves evaluation accuracy; without it, the evaluator triggers false-positive retries
- **Thinking $\times$ Tiering**: adaptive depth (within-model) and model tier routing (which-model) are complementary — two-dimensional cost optimization

---

## Reasoning Mode Ladder

| Mode | Agents | Confidence | Latency | Tokens |
|------|:------:|:----------:|:-------:|:------:|
| QUICK | 1 | 0.52 | 1.8s | 2K |
| STANDARD | 1 | 0.68 | 6.5s | 12K |
| DEEP | $N$ | 0.78 | 18.3s | 45K |
| COLLABORATIVE | $N$ | 0.82 | 24.1s | 65K |

- Standard vs Quick: $+31\%$ confidence at $6\times$ token cost
- Collaborative vs Quick: $+58\%$ confidence at $32.5\times$ token cost

The ladder provides a principled way to trade cost for quality. The deliberation layer (Algorithm 2) can select the appropriate mode based on query complexity.

---

## End-to-End Flow

```
User Query
  |
  +-- [Alg 4] Adaptive Thinking -----------> select token budget B(q)
  +-- [Alg 5] Intent Classification -------> route to sources
  |
  +-- [Alg 2] Deliberation-First
  |     |
  |     +-- RESPOND_DIRECTLY: answer from context (no tools, no cost)
  |     |
  |     +-- USE_TOOLS:
  |     |     +-- [Alg 1] ReAct Loop ---------> Think -> Act -> Observe -> Reflect
  |     |     +-- [Alg 3] Hybrid Collaborative -> Ensemble -> Blackboard -> Refine
  |     |     +-- [Alg 6] Adversarial Debate --> (if evaluative query)
  |     |
  |     +-- CLARIFY: request more information from user
  |
  +-- [Alg 7] Ensemble Aggregation --------> merge multi-agent results
  +-- [Confidence] Quality Gate ------------> C(r,p) >= 0.6 or refine
  +-- [MMR Memory] Semantic Retrieval ------> diversity-aware context
  |
  +-- Final Response
```

The deliberation layer acts as a **cost gate**: simple queries get direct answers (saving 40-60% tokens), while complex queries flow through the full multi-agent pipeline. The adaptive thinking budget ensures even within the pipeline, each component receives only the reasoning depth it needs.

### The Seven-Step Cognitive Chain

$$
\text{Observe} \to \text{Recall} \to \text{Reason} \to \text{Plan} \to \text{Act} \to \text{Reflect} \to \text{Respond}
$$

This chain mirrors human expert reasoning: perceive the problem, recall relevant context, reason about approach, plan actions, execute, self-critique, and deliver. Each algorithm in DOVA maps to one or more steps in this chain.
