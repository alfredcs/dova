# DOVA v1.4 Release Notes

**Release Date:** February 3, 2026

## Overview

DOVA v1.4 introduces an interactive CLI mode (`dova interact`) that provides a Claude Code-like continuous interaction experience with chain-of-thought reasoning, memory integration, and automatic tool selection. This release also activates the Bull vs Bear debate agents for balanced analysis.

---

## New Features

### Interactive CLI Mode (`dova interact`)

A new interactive mode that enables continuous multi-turn conversations with sophisticated reasoning:

```bash
dova interact              # Start interactive session
dova interact --no-thinking  # Hide chain-of-thought display
dova interact --verbose      # Show verbose output
```

**Capabilities:**
- **Chain-of-Thought Reasoning**: Transparent 7-step reasoning process
  1. Observe - Understand the input and context
  2. Recall - Retrieve relevant memories
  3. Reason - Chain-of-thought analysis
  4. Plan - Determine best action
  5. Act - Execute action if needed
  6. Reflect - Evaluate result and learn
  7. Respond - Generate final response

- **Memory Integration**:
  - Short-term memory (24h TTL) for session continuity
  - Long-term memory for significant research findings
  - Semantic search to recall relevant past interactions

- **Automatic Tool Selection**:
  - `research` - Search papers, repos, models for learning queries
  - `debate` - Run Bull vs Bear analysis for evaluation queries
  - `synthesize` - Combine information from multiple interactions
  - `respond` - Direct response for simple questions

- **Session Commands**:
  - `/help` - Show available commands
  - `/status` - Display session statistics
  - `/clear` - Clear conversation history
  - `/thinking on|off` - Toggle reasoning display
  - `/history` - View conversation history
  - `/memory` - Show memory references

### Bull vs Bear Debate Activation

The debate agents are now initialized on `dova serve` startup:

- **BullAgent**: Advocates for positive aspects, strengths, and opportunities
- **BearAgent**: Provides critical analysis, identifies risks and concerns
- **DebateAgent**: Orchestrates multi-round debates and synthesizes balanced conclusions

**API Endpoint:** `POST /api/v1/debate`

```json
{
  "topic": "Should organizations adopt multi-agent LLM systems?",
  "context": {"use_case": "Research automation"},
  "num_rounds": 2
}
```

**Response includes:**
- `summary` - Balanced conclusion
- `bull_strengths` - Top positive arguments
- `bear_concerns` - Top critical concerns
- `recommendation` - Actionable guidance
- `confidence_score` - Assessment confidence (0.0-1.0)
- `debate_history` - Full argument transcript

---

## Architecture

### Interactive Session Flow

```
User Input
    ↓
┌─────────────┐
│   Observe   │ → Classify intent, understand context
└─────────────┘
    ↓
┌─────────────┐
│   Recall    │ → Search short-term & long-term memory
└─────────────┘
    ↓
┌─────────────┐
│   Reason    │ → Chain-of-thought analysis
└─────────────┘
    ↓
┌─────────────┐
│    Plan     │ → Select action (research/debate/respond)
└─────────────┘
    ↓
┌─────────────┐
│    Act      │ → Execute tool if needed
└─────────────┘
    ↓
┌─────────────┐
│   Reflect   │ → Evaluate result
└─────────────┘
    ↓
┌─────────────┐
│   Respond   │ → Generate final response
└─────────────┘
    ↓
┌─────────────┐
│  Remember   │ → Store in memory for future
└─────────────┘
```

### Memory Architecture

| Type | TTL | Use Case |
|------|-----|----------|
| Short-term | 24 hours | Session continuity, recent context |
| Long-term | Persistent | Research findings, important insights |

Memory uses:
- Amazon Titan embeddings for semantic search
- MMR (Maximal Marginal Relevance) for diverse results
- Cosine similarity with configurable threshold

---

## Files Changed

### New Files

| File | Description |
|------|-------------|
| `src/dova/cli/__init__.py` | CLI package initialization |
| `src/dova/cli/interact.py` | Interactive session implementation |
| `src/dova/cli/main.py` | Main CLI commands (moved from cli.py) |

### Modified Files

| File | Changes |
|------|---------|
| `src/dova/api/main.py` | Added DebateAgent initialization |
| `src/dova/__init__.py` | Version bump to 1.4.0 |

---

## Usage Examples

### Interactive Research Session

```bash
$ dova interact

> What are the latest advances in multi-agent LLM systems?
[Observation] User asking about recent developments in multi-agent LLM...
[Reasoning] This is a technical research query requiring current information...
[Plan] Action: research

DOVA: Based on recent research, multi-agent LLM systems have seen advances in:
- Orchestration frameworks (AutoGen, CrewAI, LangGraph)
- Memory architectures for long-term context
- Tool use and function calling...

> How do they compare to single-agent approaches?
[Memory] Found 1 relevant memory from previous turn
[Plan] Action: debate

DOVA: Here's a balanced analysis:
Bull: Multi-agent enables task decomposition, specialized expertise...
Bear: Introduces coordination complexity, higher latency...
```

### Debate Analysis

```bash
$ curl -X POST "http://localhost:8081/api/v1/debate?topic=RAG%20vs%20Fine-tuning"

{
  "summary": "Both approaches have valid use cases...",
  "bull_strengths": ["No training required", "Real-time knowledge updates"...],
  "bear_concerns": ["Retrieval quality dependency", "Latency overhead"...],
  "recommendation": "Use RAG for dynamic knowledge, fine-tuning for specialized tasks",
  "confidence_score": 0.82
}
```

---

## Configuration

No new environment variables required. Interactive mode uses existing configuration:

- `BEDROCK_*` - LLM models for reasoning
- `TAVILY_API_KEY` - Web search for research actions
- Memory settings from `memory_enhanced` section

---

## Multi-Turn Conversation Intelligence

The interactive mode includes sophisticated context tracking:

### Follow-Up Response Handling
Short responses like "yes", "ok", "please" are automatically expanded using conversation context:
- Affirmative responses trigger pending suggestions
- Questions asked to the user are remembered for follow-up
- Topic continuity is maintained across turns

### Entity Tracking
Research results are stored for follow-up queries:
- Papers: title, authors, abstract, arxiv_id
- Repositories: name, description, url, stars
- Models: id, downloads
- Debate results: summary, recommendation

This enables questions like "who are the authors?" after discussing a paper.

### Context-Aware Planning
The planner distinguishes between:
- New research queries → triggers search
- Follow-up questions → uses existing context
- Synthesis requests → combines multiple turns

---

## Known Issues

- Long thinking displays add latency to response time
- Long research queries may timeout with default settings

---

## What's Next (v1.5 Roadmap)

- Streaming responses in interactive mode
- Export conversation history to Markdown
- Custom tool registration
- Multi-user session support
- Voice input/output integration

---

## Contributors

- DOVA Team

---

## Changelog Summary

| Category | Changes |
|----------|---------|
| Features | Interactive CLI mode, Debate agent activation |
| CLI | New `dova interact` command with session management |
| Memory | Enhanced recall with dataclass support |
| Architecture | CLI refactored to package structure |
