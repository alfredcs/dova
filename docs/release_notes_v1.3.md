# DOVA v1.3 Release Notes

**Release Date:** February 3, 2026

## Overview

DOVA v1.3 introduces a browser-based research UI, intelligent query understanding, answer synthesis with confidence scoring, and iterative query refinement. This release transforms DOVA from a link aggregator into a true deep research assistant that provides direct answers to research queries.

---

## New Features

### Browser-Based Research UI

- **New Frontend Interface**: Access DOVA research via browser at `http://localhost:8081/`
- **Dark Theme Design**: Modern, responsive UI with gradient styling
- **Source Selection**: Toggle chips for ArXiv, GitHub, HuggingFace, and Web sources
- **Real-Time Results**: Loading spinner with status messages during research
- **Rich Results Display**: Organized sections for papers, repositories, models, datasets, and web results

### Intelligent Query Understanding

- **Query Type Classification**: Automatic detection of query types:
  - `technical` - Architecture, algorithms, implementations
  - `biographical` - People, education, career history
  - `factual` - Definitions, facts, statistics
  - `general` - Broad topics, exploration

- **Smart Source Routing**: Queries are routed to appropriate sources based on type:
  - Biographical queries → Web search only (avoids irrelevant ArXiv results)
  - Technical queries → ArXiv, GitHub, HuggingFace, Web
  - Factual queries → Web primarily
  - General queries → All sources

### Answer Synthesis

- **Direct Answers**: LLM-synthesized answers based on research findings
- **Source Attribution**: Answers cite the sources they were derived from
- **Markdown Formatting**: Answers support bold, lists, and structured formatting
- **Prominent Display**: Research answer card displayed at top of results

### Confidence Scoring & Answer Critique

- **Confidence Scores**: Each answer receives a confidence score (0.0-1.0)
  - High confidence (≥70%): Green badge
  - Medium confidence (40-70%): Yellow badge
  - Low confidence (<40%): Red badge

- **Answer Critique**: Internal evaluation of answer quality:
  - Checks if answer addresses the query
  - Identifies missing information
  - Suggests query refinements when needed

### Iterative Query Refinement

- **Automatic Refinement**: When confidence is below threshold (70%), DOVA:
  1. Analyzes what information is missing
  2. Generates a refined search query
  3. Re-executes research with improved query
  4. Repeats up to 2 times for better results

- **Refinement Tracking**: UI displays "Query refined N times" when applicable

### Enhanced Memory Integration

- **Short-Term Memory**: Research results stored with 24-hour TTL
- **Long-Term Memory**: High-confidence (≥70%) answers stored persistently
- **Embedding-Based Search**: Semantic search using Amazon Titan embeddings
- **Memory Activation on Startup**: EnhancedMemoryService initialized automatically with `dova serve`

---

## Improvements

### Embedding Model Fix

- **Issue**: Embedding generation was failing with "messages: Field required"
- **Root Cause**: Claude models were being used for embeddings (Claude doesn't support embeddings)
- **Fix**: Bedrock provider now uses Amazon Titan (`amazon.titan-embed-text-v2:0`) for embedding tasks
- **Result**: Memory embeddings now generate successfully

### Web Search Enhancement

- **Tavily Integration**: Added `tavily-python>=0.5.0` as a dependency
- **Improved Coverage**: Better web search results for non-technical queries

### Response Schema Updates

New fields in `ResearchResponse`:
- `answer` (str): Direct synthesized answer
- `confidence` (float): Answer confidence score (0.0-1.0)
- `refinement_attempts` (int): Number of query refinements performed
- `datasets` (list): HuggingFace datasets found
- `web_results` (list): Web search results

---

## API Changes

### Research Endpoint Updates

The `/api/v1/research` endpoint now returns:

```json
{
  "query": "original query",
  "status": "completed",
  "answer": "Synthesized answer based on research...",
  "confidence": 0.85,
  "refinement_attempts": 0,
  "summary": "Found 5 papers, 3 repositories...",
  "papers": [...],
  "repositories": [...],
  "models": [...],
  "datasets": [...],
  "web_results": [...],
  "insights": [...],
  "recommendations": [...],
  "metadata": {
    "execution_time_ms": 2500,
    "sources_searched": ["arxiv", "github", "web"]
  }
}
```

### New Frontend Route

- `GET /` - Serves the research UI (index.html)
- `GET /static/*` - Static assets (CSS, JS, images)

---

## Configuration

### New Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BEDROCK_EMBEDDING_MODEL` | Amazon Titan model for embeddings | `amazon.titan-embed-text-v2:0` |
| `TAVILY_API_KEY` | Tavily API key for enhanced web search | (optional) |

---

## Dependencies

### Added
- `tavily-python>=0.5.0` - Tavily web search client

### Updated
- No dependency version changes

---

## Migration Guide

### From v1.2 to v1.3

1. **Install new dependencies**:
   ```bash
   pip install -e ".[dev]"
   # or
   pip install tavily-python>=0.5.0
   ```

2. **Optional: Configure Tavily** (for enhanced web search):
   ```bash
   # Add to .env
   TAVILY_API_KEY=your-tavily-api-key
   ```

3. **Access the new UI**:
   ```bash
   dova serve --port 8081
   # Open http://localhost:8081 in browser
   ```

4. **Embedding model**: No action needed - automatically uses Amazon Titan

---

## Known Issues

- **Anonymous Access Warning**: In development mode, requests show "anonymous_access" warning. This is expected behavior and not an error.

---

## What's Next (v1.4 Roadmap)

- Memory-assisted query improvement using past research
- Multi-turn research conversations
- Export research results to PDF/Markdown
- Collaborative research sessions
- Citation management and bibliography generation

---

## Contributors

- DOVA Team

---

## Changelog Summary

| Category | Changes |
|----------|---------|
| Features | Frontend UI, Query Understanding, Answer Synthesis, Confidence Scoring, Iterative Refinement |
| Fixes | Embedding model (Titan), Tavily dependency |
| API | New response fields (answer, confidence, refinement_attempts) |
| Docs | Updated README, Getting Started, Release Notes |
