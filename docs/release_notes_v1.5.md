# DOVA v1.5 Release Notes

**Release Date:** February 4, 2026

## Overview

DOVA v1.5 introduces the **ThinkingOrchestrator** - a deliberation-first orchestration approach that reasons about user needs before deciding which tools to use. This release shifts from predetermined task-graph orchestration to intelligent, context-aware decision making.

---

## New Features

### ThinkingOrchestrator (Deliberation-First Orchestration)

A new orchestrator that thinks before acting, providing smarter tool selection and personalized responses:

```bash
# CLI usage
dova interact --orchestrator thinking
dova research "explain attention mechanisms" --orchestrator thinking

# In interactive mode
/orchestrator thinking
```

**Key Innovations:**

- **Deliberation Before Action**: The orchestrator explicitly reasons about what the user needs before deciding which tools (if any) to invoke
- **Context-Aware Tool Selection**: ArXiv, GitHub, HuggingFace, and web search are only called when reasoning determines they would help
- **User Model Integration**: Deep understanding of user expertise, communication preferences, and goals
- **Conversation Memory**: Rich session context for follow-up questions without re-searching

### How It Works

**Traditional Flow (Task-Graph):**
```
Query -> Intent Classification -> Build Task Graph -> Execute All Sources -> Synthesize
```

**New Flow (Deliberation-First):**
```
Query -> Gather Context -> DELIBERATE (think) -> Decide Action -> Execute Only If Needed -> Personalize Response
```

### Deliberation Process

The ThinkingOrchestrator uses structured reasoning:

```json
{
  "understanding": "User wants current news about EU AI policy",
  "can_answer_from_context": false,
  "knowledge_gaps": ["latest regulatory updates"],
  "tools_to_use": [
    {"tool": "web", "rationale": "Current events, not academic research", "query": "EU AI regulation 2026"}
  ],
  "action": "use_tools",
  "reasoning": "This is a news query - ArXiv/GitHub/HuggingFace won't help"
}
```

**Action Types:**
| Action | When Used |
|--------|-----------|
| `respond_directly` | Answer exists in context or knowledge |
| `use_tools` | External information needed |
| `clarify` | Query is ambiguous |

### User Model

Rich user representation for personalized orchestration:

```python
@dataclass
class UserModel:
    user_id: str
    expertise_areas: dict[str, ExpertiseLevel]  # e.g., {"transformers": EXPERT}
    preferred_depth: ResponseDepth  # BRIEF, STANDARD, DETAILED
    prefers_code_examples: bool
    prefers_citations: bool
    formality: str  # "technical", "casual", "formal"
    current_goals: list[str]
    entities_of_interest: dict[str, Any]
```

**Expertise Levels:**
- `BEGINNER` - New to topic
- `INTERMEDIATE` - Working knowledge
- `EXPERT` - Deep expertise
- `UNKNOWN` - Not yet determined

### Conversation Context

Session memory for multi-turn conversations:

```python
@dataclass
class ConversationContext:
    session_id: str
    turns: list[ConversationTurn]
    current_topic: str
    papers_discussed: list[dict]
    repos_discussed: list[dict]
    models_discussed: list[dict]
    topic_history: list[str]
```

**Features:**
- Entity tracking across turns
- Reference resolution ("the first paper", "that repo")
- Topic continuity
- Deduplication of discussed entities

---

## Architecture

### ThinkingOrchestrator Flow

```
User Query
    ↓
┌─────────────────────┐
│  Load User Model    │ → Expertise, preferences, goals
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Load Context       │ → Session history, entities discussed
└─────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│              DELIBERATE                  │
│  - What does user actually need?         │
│  - Can I answer from context?            │
│  - Which tools would help (if any)?      │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────┐
│  Execute Decision   │ → respond_directly / use_tools / clarify
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Personalize        │ → Adapt to user expertise/style
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Update Context     │ → Store for future turns
└─────────────────────┘
```

### Comparison with Standard Orchestrator

| Aspect | Standard (DOVAOrchestrator) | Thinking (ThinkingOrchestrator) |
|--------|----------------------------|--------------------------------|
| Tool Selection | Predetermined by query type | Deliberated per query |
| User Awareness | Basic profiling | Rich user model |
| Context | Session-based | Entity-aware conversation |
| Follow-ups | Re-search sources | Use existing context |
| Response Style | Generic | Personalized to expertise |

---

## Files Changed

### New Files

| File | Description |
|------|-------------|
| `src/dova/agents/thinking_orchestrator.py` | ThinkingOrchestrator implementation (~500 lines) |
| `src/dova/agents/user_model.py` | UserModel, ExpertiseLevel, ResponseDepth classes |
| `src/dova/agents/conversation_context.py` | ConversationContext, ConversationTurn classes |
| `tests/unit/agents/test_thinking_orchestrator.py` | 25 unit tests for new components |

### Modified Files

| File | Changes |
|------|---------|
| `src/dova/agents/profiling.py` | Added `to_user_model()` conversion method |
| `src/dova/api/schemas/chat.py` | Added `orchestrator` field to ChatRequest |
| `src/dova/api/routes/chat.py` | Pass orchestrator_type to session creation |
| `src/dova/cli/main.py` | Added `--orchestrator` option to interact/research commands |
| `src/dova/cli/interact.py` | Added ThinkingOrchestrator support, `/orchestrator` command |
| `frontend/src/api/types.ts` | Added orchestrator to ResearchQuery |
| `frontend/src/components/search/SearchFilters.tsx` | Added orchestrator mode selection UI |
| `frontend/src/pages/Dashboard.tsx` | Added orchestrator state management |

---

## Usage Examples

### CLI: Interactive Mode with Thinking Orchestrator

```bash
$ dova interact --orchestrator thinking

> What's the latest on EU AI regulation?
[Deliberation] This is a current events query about policy...
[Decision] Use web search only (not academic sources)
[Tool] web: "EU AI regulation 2026"

DOVA: The EU AI Act came into force in August 2024...

> Who proposed it?
[Deliberation] Follow-up about entity from previous turn...
[Decision] Answer from context (no tools needed)

DOVA: The EU AI Act was proposed by the European Commission...
```

### CLI: Research Command

```bash
# Traditional orchestration
dova research "transformer architecture" --orchestrator standard

# Deliberation-first orchestration
dova research "explain attention" --orchestrator thinking
```

### Switching Orchestrators in Interactive Mode

```bash
$ dova interact

> /orchestrator thinking
Switched to thinking orchestrator (deliberation-first)

> /orchestrator standard
Switched to standard orchestrator (task-graph)

> /orchestrator
Current orchestrator: thinking
```

### API Usage

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is BERT?",
    "orchestrator": "thinking"
  }'
```

### Frontend UI

The search interface now includes an orchestrator mode selector:
- **Standard**: Task-graph orchestration (default)
- **Thinking**: Deliberation-first (experimental)

---

## Configuration

No new environment variables required. ThinkingOrchestrator uses existing LLM configuration.

**Defaults:**
- CLI: `--orchestrator standard` (backwards compatible)
- API: `orchestrator: "standard"` (backwards compatible)
- Interactive: Can switch with `/orchestrator` command

---

## Example Behaviors

### Query: "What's the latest on EU AI regulation?"

**Standard Orchestrator:**
- Classifies as RESEARCH_QUERY
- Searches ArXiv, GitHub, HuggingFace, Web in parallel
- Returns results from all sources (many irrelevant)

**Thinking Orchestrator:**
- Deliberates: "User wants current news about policy"
- Decides: Only web search is relevant
- Executes: Web search only
- Result: Focused, relevant response

### Query: "Who are the authors?" (follow-up)

**Standard Orchestrator:**
- Classifies as new RESEARCH_QUERY
- Re-searches all sources

**Thinking Orchestrator:**
- Deliberates: "Follow-up about paper from previous turn"
- Decides: Answer from conversation context
- Executes: No tools called
- Result: Instant response from memory

---

## Known Limitations

- Deliberation adds slight latency (~1-2 seconds for LLM reasoning)
- User model is session-scoped (not persisted across sessions yet)
- Limited to 4 tool types (arxiv, github, huggingface, web)

---

## What's Next (v1.6 Roadmap)

- Persistent user models across sessions
- Learning from user feedback
- Custom tool registration for ThinkingOrchestrator
- Streaming deliberation display
- Multi-modal context (images, code snippets)

---

## Contributors

- DOVA Team

---

## Changelog Summary

| Category | Changes |
|----------|---------|
| Features | ThinkingOrchestrator, UserModel, ConversationContext |
| CLI | `--orchestrator` option, `/orchestrator` command |
| API | `orchestrator` field in ChatRequest |
| Frontend | Orchestrator mode selector in search filters |
| Tests | 25 new unit tests for thinking orchestrator |
