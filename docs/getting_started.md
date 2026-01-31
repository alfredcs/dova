# DOVA Getting Started Guide

This guide will walk you through setting up, configuring, and using the DOVA (Deep Orchestrated Versatile Agent Platform) for AI/ML research automation.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running Locally](#running-locally)
5. [Using the CLI](#using-the-cli)
6. [API Usage](#api-usage)
7. [Agentic Reasoning](#agentic-reasoning)
8. [Managing Custom Sources](#managing-custom-sources)
9. [Proactive Recommendations](#proactive-recommendations)
10. [Sandbox Execution](#sandbox-execution)
11. [Architecture Overview](#architecture-overview)
12. [Deployment](#deployment)
13. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before installing DOVA, ensure you have the following:

### Required

- **Python 3.11+** - DOVA requires Python 3.11 or higher
- **AWS Account** - For Bedrock access and deployment
- **AWS CLI** - Configured with appropriate credentials

### Optional (but recommended)

- **Docker** - For local Redis and containerized deployment
- **Node.js 18+** - For CDK infrastructure deployment
- **uv** - Fast Python package manager (recommended over pip)

### Verify Prerequisites

```bash
# Check Python version
python3 --version  # Should be 3.11+

# Check AWS CLI
aws --version
aws sts get-caller-identity  # Verify credentials

# Check Docker (optional)
docker --version

# Check Node.js (optional)
node --version  # Should be 18+
```

---

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/dova.git
cd dova

# Run the setup script
./scripts/setup.sh
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with uv (recommended)
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Check DOVA is installed
dova --version

# Run basic tests
pytest tests/unit -v
```

---

## Configuration

### Environment Variables

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```bash
# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO

# AWS Configuration
AWS_REGION=us-east-1
AWS_PROFILE=default

# LLM Configuration
LLM_PRIMARY_PROVIDER=bedrock
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# Optional: Direct API access (fallback)
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key

# MCP Servers
MCP_ENABLED_SERVERS=arxiv,github,huggingface

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Redis (required for caching and job queue)
REDIS_HOST=localhost
REDIS_PORT=6379

# Background Jobs
JOB_WORKER_CONCURRENCY=5
JOB_ARXIV_POLL_HOURS=1.0
JOB_HF_POLL_HOURS=6.0

# Sandbox Execution (optional)
SANDBOX_ENABLED=false
SANDBOX_DOCKER_HOST=unix:///var/run/docker.sock
SANDBOX_NETWORK_ENABLED=false
SANDBOX_MAX_CONCURRENT=5
```

### AWS Bedrock Setup

1. **Enable Bedrock Models** in your AWS account:
   - Go to AWS Console → Amazon Bedrock → Model access
   - Request access to Claude models (Sonnet recommended)

2. **Configure IAM Permissions**:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "bedrock:InvokeModel",
           "bedrock:InvokeModelWithResponseStream"
         ],
         "Resource": "arn:aws:bedrock:*::foundation-model/*"
       }
     ]
   }
   ```

### MCP Server Configuration

DOVA integrates with Model Context Protocol (MCP) servers for external data sources. Configuration is stored in `~/.dova.json` (similar to `~/.claude.json`).

#### Quick Setup

```bash
# Add ArXiv server (no auth required)
dova mcp add arxiv --url http://infs.cavatar.info:8084/mcp

# Add HuggingFace server
dova mcp add huggingface --url https://huggingface.co/mcp

# Add GitHub server with token (pass token directly with -H)
dova mcp add github --url https://api.githubcopilot.com/mcp -H ghp_yourtoken

# List configured servers
dova mcp list

# Test a server
dova mcp test huggingface --tool model_search
```

#### Configuration File

MCP servers are stored in `~/.dova.json`:

```json
{
  "mcpServers": {
    "arxiv": {
      "type": "http",
      "url": "http://infs.cavatar.info:8084/mcp"
    },
    "huggingface": {
      "type": "http",
      "url": "https://huggingface.co/mcp"
    },
    "github": {
      "type": "http",
      "url": "https://api.githubcopilot.com/mcp",
      "headers": {
        "Authorization": "Bearer ghp_yourtoken"
      }
    }
  }
}
```

#### Server Name Requirements

Server names must match what the research agent expects:

| Server Name | Purpose | Required |
|-------------|---------|----------|
| `arxiv` | Academic papers | Yes (for paper search) |
| `github` | Code repositories | Yes (for code search) |
| `huggingface` | ML models/datasets | Yes (for model search) |

#### MCP CLI Commands

| Command | Description |
|---------|-------------|
| `dova mcp add <name> --url <url>` | Add or update an MCP server |
| `dova mcp add <name> --url <url> -H <token>` | Add server with Bearer auth token |
| `dova mcp add <name> --url <url> -H "Key: Value"` | Add server with custom header |
| `dova mcp remove <name>` | Remove an MCP server |
| `dova mcp list` | List all configured servers |
| `dova mcp show` | Show full config file |
| `dova mcp test <name>` | Test server connectivity |

### Custom Sources

Beyond built-in MCP servers, users can add custom sources through the API:

| Source Type | Description | Example |
|-------------|-------------|---------|
| Web URL | Scrape web pages | Documentation sites, blogs |
| RSS Feed | Parse RSS/Atom feeds | News feeds, publication updates |
| Custom API | Call HTTP APIs | Internal search APIs, third-party services |

Custom sources support authentication (Bearer tokens, API keys) and learn quality scores from user interactions. See [Managing Custom Sources](#managing-custom-sources) for API details.

---

## Running Locally

### Start the API Server

```bash
# Using the CLI
dova serve

# Or with uvicorn directly
uvicorn dova.api.main:app --reload --host 0.0.0.0 --port 8000

# Or using Make
make run-local
```

### Start with Docker Compose

```bash
# Start all services (API + Redis)
docker-compose up -d

# Start with background workers (for proactive recommendations)
docker-compose up -d api redis worker

# View logs
docker-compose logs -f api

# View worker logs
docker-compose logs -f worker

# Stop services
docker-compose down
```

### Verify the Server

```bash
# Check health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "version": "0.1.0", "environment": "development"}
```

---

## Using the CLI

DOVA provides a command-line interface for common operations.

### Research Queries

```bash
# Basic research query
dova research "transformer architecture for NLP"

# Specify sources
dova research "reinforcement learning" -s arxiv -s github

# Limit results
dova research "BERT models" -n 5

# Save to file
dova research "attention mechanisms" -o results.json -f json
```

### Code Validation

```bash
# Validate a Python file
dova validate ./src/my_module.py

# Specify language
dova validate ./main.go -l go

# Save report
dova validate ./project -o report.json
```

### System Health

```bash
# Check system status
dova health

# Check with exit code (for scripts)
dova health --check
```

### Configuration

```bash
# Show current configuration
dova config
```

### User Profiles

```bash
# Show user profile
dova profile show user-123

# Update profile
dova profile update user-123 -i "machine learning" -i "NLP" -e advanced
```

### MCP Server Management

Manage MCP server configurations stored in `~/.dova.json`:

```bash
# Add MCP servers
dova mcp add arxiv --url http://infs.cavatar.info:8084/mcp
dova mcp add huggingface --url https://huggingface.co/mcp
dova mcp add github --url https://api.githubcopilot.com/mcp -H ghp_yourtoken

# List configured servers
dova mcp list

# Show full configuration
dova mcp show

# Test server connectivity
dova mcp test huggingface
dova mcp test huggingface --tool model_search

# Remove a server
dova mcp remove arxiv
```

The `-H` flag accepts either:
- **Token shorthand**: `-H ghp_token` → becomes `Authorization: Bearer ghp_token`
- **Full header**: `-H "X-Api-Key: mykey"` → becomes `X-Api-Key: mykey`

---

## API Usage

### Authentication

DOVA supports two authentication methods:

1. **JWT Token** (for user applications):
   ```bash
   curl -H "Authorization: Bearer <jwt-token>" \
        http://localhost:8000/api/v1/research
   ```

2. **API Key** (for programmatic access):
   ```bash
   curl -H "X-API-Key: <api-key>" \
        http://localhost:8000/api/v1/research
   ```

### Research Endpoint

```bash
# POST /api/v1/research
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "query": "latest advances in large language models",
    "sources": ["arxiv", "github", "huggingface"],
    "max_results": 10,
    "include_synthesis": true
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "summary": "Recent LLM advances focus on...",
    "papers": [
      {
        "title": "Attention Is All You Need",
        "id": "1706.03762",
        "authors": ["Vaswani et al."],
        "relevance_score": 0.95
      }
    ],
    "repositories": [...],
    "models": [...],
    "insights": [...],
    "recommendations": [...]
  },
  "metadata": {
    "query_time_ms": 2500,
    "sources_queried": ["arxiv", "github", "huggingface"]
  }
}
```

### Search Endpoints

```bash
# Search ArXiv papers
curl -X POST http://localhost:8000/api/v1/search/arxiv \
  -H "Content-Type: application/json" \
  -d '{"query": "neural networks", "max_results": 10}'

# Search GitHub repositories
curl -X POST http://localhost:8000/api/v1/search/github \
  -d '{"query": "pytorch transformers", "max_results": 10}'

# Search HuggingFace models
curl -X POST http://localhost:8000/api/v1/search/huggingface \
  -d '{"query": "text generation", "max_results": 10}'
```

### Profile Management

```bash
# Get user profile
curl http://localhost:8000/api/v1/profile \
  -H "Authorization: Bearer <token>"

# Update profile
curl -X PUT http://localhost:8000/api/v1/profile \
  -H "Authorization: Bearer <token>" \
  -d '{
    "interests": ["deep learning", "computer vision"],
    "expertise_level": "advanced"
  }'
```

### Code Validation

```bash
# Static analysis (always available)
curl -X POST http://localhost:8000/api/v1/validate \
  -H "Authorization: Bearer <token>" \
  -d '{
    "code": "def hello(): print(\"world\")",
    "language": "python"
  }'

# Execute code in sandbox (requires SANDBOX_ENABLED=true)
curl -X POST http://localhost:8000/api/v1/validate/execute \
  -H "Authorization: Bearer <token>" \
  -d '{
    "code": "print(2 + 2)",
    "language": "python"
  }'

# Check execution quota
curl http://localhost:8000/api/v1/validate/quota \
  -H "Authorization: Bearer <token>"
```

### Python SDK Usage

```python
import httpx
from dova.api.schemas.research import ResearchRequest

async def research_query():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/research",
            json={
                "query": "transformer models",
                "sources": ["arxiv", "github"],
                "max_results": 5,
            },
            headers={"Authorization": "Bearer <token>"}
        )
        return response.json()
```

---

## Agentic Reasoning

DOVA includes advanced reasoning capabilities that make individual agents smarter (intra-agent) and enable collaborative reasoning between agents (inter-agent) for synergistic outcomes.

### Reasoning Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `quick` | Single-pass, no reflection | Fast queries, simple lookups |
| `standard` | ReAct + self-reflection | Balanced depth, default mode |
| `deep` | Full ensemble reasoning | Complex analysis, multiple perspectives |
| `collaborative` | Blackboard + ensemble + iterative | Research synthesis, 1+1>2 outcomes |
| `tool_augmented` | Proactive tool discovery + execution | Tasks requiring external tools |

### Individual Agent Reasoning (ReasoningMixin)

Every DOVA agent inherits reasoning capabilities:

```python
from dova.agents.base import BaseAgent

class MyAgent(BaseAgent):
    async def analyze(self, problem: str):
        # ReAct-style reasoning with automatic reflection
        trace = await self.reason(
            problem=problem,
            max_iterations=5,
            reflect=True,
            available_actions=["search", "analyze", "synthesize", "conclude"]
        )

        # Access the reasoning trace
        print(f"Final answer: {trace.refined_answer}")
        print(f"Confidence: {trace.confidence}")
        print(f"Self-critique: {trace.self_critique}")

        # Working memory (scratchpad)
        self.update_scratchpad("key_insight", "important finding")
        context = self.get_scratchpad()
```

**ReAct Loop Flow:**
1. **THOUGHT**: Agent reasons about what to do next
2. **ACTION**: Executes chosen action (search, analyze, etc.)
3. **OBSERVATION**: Records what was learned
4. **REFLECTION**: Self-critiques and refines the answer

### Collaborative Reasoning

For complex problems, multiple agents can collaborate:

#### Blackboard Pattern
Agents post hypotheses, evidence, and refinements to a shared workspace:

```python
from dova.services.blackboard import Blackboard, PostType

blackboard = Blackboard()

# Agent 1 posts hypothesis
post_id = await blackboard.post(
    agent_name="ResearchAgent",
    post_type=PostType.HYPOTHESIS,
    content="Transformers outperform RNNs for long sequences",
    confidence=0.8
)

# Agent 2 adds evidence
await blackboard.post(
    agent_name="ValidationAgent",
    post_type=PostType.EVIDENCE,
    content="Benchmark shows 15% improvement on sequence length >1000",
    references=[post_id]
)

# Agent 3 votes on the hypothesis
await blackboard.vote(post_id, "SynthesisAgent", agreement=0.9, reasoning="Consistent with literature")

# Synthesize conclusions
synthesis = await blackboard.synthesize()
```

#### Ensemble Reasoning
Multiple agents solve in parallel, results are aggregated:

```python
from dova.services.ensemble import EnsembleReasoning, AggregationMethod

ensemble = EnsembleReasoning(llm_func=orchestrator.think)

result = await ensemble.reason(
    problem="What are the best practices for RAG systems?",
    agents=[research_agent, synthesis_agent, validation_agent],
    method=AggregationMethod.SYNTHESIS  # or VOTE, BEST_OF, UNION
)

print(f"Synthesized answer: {result.synthesized_answer}")
print(f"Agreement score: {result.agreement_score}")
print(f"Dissenting views: {result.dissenting_views}")
```

#### Full Collaborative Orchestration
Combines all patterns for maximum insight:

```python
from dova.services.collaborative import CollaborativeReasoning, CollaborationMode

collab = CollaborativeReasoning(llm_func=orchestrator.think)

result = await collab.reason(
    problem="Evaluate transformer vs state-space models for sequence modeling",
    agents=[research_agent, validation_agent, synthesis_agent],
    mode=CollaborationMode.HYBRID,  # Ensemble → Blackboard → Iterative
    max_iterations=3
)

print(f"Final answer: {result.final_answer}")
print(f"Confidence: {result.confidence}")
print(f"Participating agents: {result.participating_agents}")
print(f"Refinement history: {result.refinement_history}")
```

#### Tool-Augmented Reasoning
Proactively discovers and uses tools based on task analysis:

```python
from dova.services.collaborative import CollaborativeReasoning, CollaborationMode
from dova.services.tool_resolver import ToolResolver, TaskAnalyzer

# Create collaborative reasoning with tool discovery
collab = CollaborativeReasoning(
    llm_func=orchestrator.think,
    settings=settings,
    mcp_client=mcp_client,
    sandbox_executor=sandbox,
    memory_service=memory,
)

# Tool-augmented mode: analyze task → discover tools → execute → reason
result = await collab.reason(
    problem="Find papers on attention mechanisms and validate the implementations",
    agents=[research_agent, validation_agent],
    mode=CollaborationMode.TOOL_AUGMENTED,
    max_iterations=3,
)

print(f"Final answer: {result.final_answer}")
print(f"Tools used: {result.tools_used}")
print(f"Tool results: {result.tool_results}")
print(f"Tool plan: {result.tool_plan}")
```

**Tool Categories:**
| Category | Description | Example Tools |
|----------|-------------|---------------|
| `SEARCH` | Search external sources | ArXiv, GitHub, HuggingFace via MCP |
| `EXECUTE` | Code execution | Sandbox Docker containers |
| `VALIDATE` | Code analysis | Static analysis, security checks |
| `SYNTHESIZE` | Result combination | Cross-source synthesis |
| `RECOMMEND` | Personalized suggestions | Subscription-based recommendations |
| `MEMORY` | Knowledge storage | AgentCore memory recall/store |
| `WEB` | Web access | URL fetching, API calls |
| `PROFILE` | User preferences | Profile retrieval |

**Task Analysis:**
The TaskAnalyzer automatically detects required capabilities from the problem:

```python
from dova.services.tool_resolver import TaskAnalyzer

analyzer = TaskAnalyzer()
requirements = analyzer.analyze("Search for transformer papers and run the code")

print(f"Categories: {requirements.categories}")  # [SEARCH, EXECUTE]
print(f"Requires search: {requirements.requires_search}")  # True
print(f"Requires execution: {requirements.requires_execution}")  # True
print(f"Complexity: {requirements.complexity}")  # "moderate"
```

**Tool Resolution:**
The ToolResolver discovers available tools and creates execution plans:

```python
from dova.services.tool_resolver import ToolResolver

resolver = ToolResolver(settings=settings, mcp_registry=mcp_registry)

# Discover all available tools
tools = resolver.discover_tools()

# Create a plan for a specific task
plan = resolver.create_plan(
    task="Find machine learning papers and validate code",
    context={"user_id": "user123"},
)

print(f"Selected tools: {[t.name for t in plan.selected_tools]}")
print(f"Execution order: {plan.execution_order}")
print(f"Fallback tools: {[t.name for t in plan.fallback_tools]}")
```

### API Usage with Reasoning Mode

```bash
# Standard reasoning (default)
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest advances in multi-agent systems",
    "reasoning_mode": "standard"
  }'

# Deep ensemble reasoning
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "compare RAG architectures",
    "reasoning_mode": "deep"
  }'

# Full collaborative mode
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{
    "query": "synthesize best practices for LLM fine-tuning",
    "reasoning_mode": "collaborative"
  }'
```

### CLI with Reasoning

```bash
# Quick mode for fast responses
dova research "what is BERT?" --reasoning quick

# Deep reasoning for complex analysis
dova research "compare GPT-4 vs Claude architectures" --reasoning deep

# Collaborative mode for synthesis
dova research "evaluate RAG vs fine-tuning tradeoffs" --reasoning collaborative

# Tool-augmented mode for automatic tool discovery
dova research "find papers and validate implementations" --reasoning tool_augmented
```

---

## Managing Custom Sources

DOVA supports user-defined custom sources beyond the built-in ArXiv, GitHub, and HuggingFace integrations. Custom sources can be Web URLs, RSS/Atom feeds, or custom API endpoints.

### Source Types

| Type | Description | Use Case |
|------|-------------|----------|
| `web_url` | Scrapes web pages for content | Blogs, documentation sites |
| `rss_feed` | Parses RSS/Atom feeds | News sites, publication feeds |
| `api` | Calls custom API endpoints | Internal APIs, third-party services |

### Quality Learning

DOVA learns source quality from implicit usage signals:
- **Queries**: How often the source is searched
- **Clicks**: How often results are clicked
- **Saves**: How often results are saved/bookmarked

Sources with higher quality scores are prioritized in search results.

### API Endpoints

#### List Sources

```bash
# Get all sources (built-in + custom)
curl http://localhost:8000/api/v1/sources \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
[
  {
    "id": "arxiv",
    "name": "ArXiv",
    "source_type": "builtin",
    "enabled": true,
    "quality": {"query_count": 1000, "click_count": 500, "quality_score": 0.65}
  },
  {
    "id": "custom_abc123",
    "name": "ML Blog",
    "source_type": "rss_feed",
    "enabled": true,
    "quality": {"query_count": 50, "click_count": 20, "quality_score": 0.55}
  }
]
```

#### Add Custom Source

```bash
# Add an RSS feed source
curl -X POST http://localhost:8000/api/v1/sources \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Hacker News",
    "source_type": "rss_feed",
    "config": {
      "url": "https://news.ycombinator.com/rss"
    }
  }'

# Add a custom API source
curl -X POST http://localhost:8000/api/v1/sources \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Internal Docs API",
    "source_type": "api",
    "config": {
      "url": "https://api.example.com/search?q={query}",
      "auth_type": "bearer",
      "auth_value": "your-api-token"
    }
  }'
```

#### Update Source

```bash
# Disable a source
curl -X PUT http://localhost:8000/api/v1/sources/custom_abc123 \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'
```

#### Delete Source

```bash
curl -X DELETE http://localhost:8000/api/v1/sources/custom_abc123 \
  -H "Authorization: Bearer <token>"
```

#### Record Interaction (for quality learning)

```bash
# Record a click on a search result
curl -X POST http://localhost:8000/api/v1/sources/interact \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "source_id": "custom_abc123",
    "interaction_type": "click",
    "result_position": 2
  }'
```

### Source Configuration Options

| Field | Type | Description |
|-------|------|-------------|
| `url` | string | Source URL (use `{query}` placeholder for search term) |
| `headers` | object | Custom HTTP headers |
| `auth_type` | string | Authentication type: `bearer`, `api_key`, or `basic` |
| `auth_value` | string | Authentication credential |
| `content_selector` | string | CSS selector for web scraping (web_url only) |
| `refresh_interval_minutes` | int | Cache refresh interval (default: 60) |

---

## Proactive Recommendations

DOVA proactively monitors ArXiv and HuggingFace for new content matching your interests, delivering personalized recommendations without requiring manual searches.

### How It Works

1. **Background Polling**: Workers periodically poll ArXiv (hourly) and HuggingFace (every 6 hours)
2. **Content Processing**: New items are normalized and embedded for matching
3. **User Matching**: Content is matched to users based on subscriptions and profile similarity
4. **Delivery**: Recommendations are batched and delivered with frequency capping

### Managing Subscriptions

Subscribe to topics, authors, or categories to receive proactive recommendations:

```bash
# List your subscriptions
curl http://localhost:8000/api/v1/subscriptions \
  -H "Authorization: Bearer <token>"

# Subscribe to an ArXiv category
curl -X POST http://localhost:8000/api/v1/subscriptions \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "arxiv_category",
    "value": "cs.AI"
  }'

# Subscribe to ArXiv keyword
curl -X POST http://localhost:8000/api/v1/subscriptions \
  -H "Authorization: Bearer <token>" \
  -d '{"type": "arxiv_keyword", "value": "transformer"}'

# Subscribe to HuggingFace task
curl -X POST http://localhost:8000/api/v1/subscriptions \
  -H "Authorization: Bearer <token>" \
  -d '{"type": "hf_task", "value": "text-generation"}'

# Delete a subscription
curl -X DELETE http://localhost:8000/api/v1/subscriptions/<sub-id> \
  -H "Authorization: Bearer <token>"
```

### Subscription Types

| Type | Description | Example Values |
|------|-------------|----------------|
| `arxiv_category` | ArXiv subject categories | `cs.AI`, `cs.LG`, `cs.CL` |
| `arxiv_author` | ArXiv paper authors | `Vaswani`, `Hinton` |
| `arxiv_keyword` | Keywords in paper titles/abstracts | `transformer`, `attention` |
| `hf_task` | HuggingFace model tasks | `text-generation`, `image-classification` |
| `hf_author` | HuggingFace model authors | `google`, `meta-llama` |
| `github_repo` | GitHub repositories | `huggingface/transformers` |
| `github_topic` | GitHub topics | `machine-learning`, `nlp` |

### Getting Recommendations

```bash
# Get personalized recommendations
curl http://localhost:8000/api/v1/subscriptions/recommendations \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "recommendations": [
    {
      "content_id": "arxiv:2401.12345",
      "source": "arxiv",
      "title": "Attention Improvements for Long Sequences",
      "url": "https://arxiv.org/abs/2401.12345",
      "score": 0.92,
      "matched_tags": ["transformer", "attention"],
      "reason": "Matches your interests in transformer, attention"
    }
  ],
  "total": 15,
  "page": 1,
  "page_size": 10
}
```

### Delivery Preferences

Control how recommendations are delivered:

```bash
# Get delivery preferences
curl http://localhost:8000/api/v1/subscriptions/preferences \
  -H "Authorization: Bearer <token>"

# Update preferences
curl -X PATCH http://localhost:8000/api/v1/subscriptions/preferences \
  -H "Authorization: Bearer <token>" \
  -d '{
    "max_daily": 20,
    "min_score": 0.8,
    "batch_size": 5,
    "cooldown_hours": 6
  }'
```

| Preference | Description | Default |
|------------|-------------|---------|
| `max_daily` | Maximum recommendations per day | 10 |
| `min_score` | Minimum relevance score (0-1) | 0.75 |
| `batch_size` | Max items per notification | 5 |
| `cooldown_hours` | Hours between notifications | 4 |

### GitHub Webhooks

Receive real-time updates from GitHub repositories:

```bash
# Configure webhook in GitHub repository settings:
# Payload URL: https://your-domain/api/v1/webhooks/github
# Content type: application/json
# Events: push, release, star, issues
```

---

## Sandbox Execution

DOVA provides secure, isolated code execution in Docker containers with resource tiers and quota management.

### Enabling Sandbox

Set the following environment variables:

```bash
# Enable sandbox execution
SANDBOX_ENABLED=true

# Docker socket (default)
SANDBOX_DOCKER_HOST=unix:///var/run/docker.sock

# Optional: Allow network access in sandbox
SANDBOX_NETWORK_ENABLED=false

# Concurrency limit
SANDBOX_MAX_CONCURRENT=5
```

### Resource Tiers

Code is automatically assigned to a tier based on analysis, or you can specify explicitly:

| Tier | CPU | Memory | Timeout | Use Case |
|------|-----|--------|---------|----------|
| `cpu_basic` | 0.5 vCPU | 512MB | 60s | Simple scripts, quick tests |
| `cpu_standard` | 2 vCPU | 4GB | 300s | Data processing, medium workloads |
| `gpu_spot` | 4 vCPU + T4 GPU | 16GB | 600s | ML inference, training |
| `gpu_premium` | 8 vCPU + A10 GPU | 32GB | 1800s | Large model training |

**Automatic Tier Detection**: DOVA analyzes your code for imports like `torch`, `tensorflow`, `numpy` to automatically select the appropriate tier.

### Executing Code

```bash
# Execute Python code
curl -X POST http://localhost:8000/api/v1/validate/execute \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import math\nprint(math.sqrt(16))",
    "language": "python"
  }'

# With dependencies
curl -X POST http://localhost:8000/api/v1/validate/execute \
  -H "Authorization: Bearer <token>" \
  -d '{
    "code": "import numpy as np\nprint(np.array([1,2,3]).sum())",
    "language": "python",
    "dependencies": ["numpy"]
  }'

# Specify tier explicitly
curl -X POST http://localhost:8000/api/v1/validate/execute \
  -H "Authorization: Bearer <token>" \
  -d '{
    "code": "import torch\nprint(torch.cuda.is_available())",
    "language": "python",
    "tier": "gpu_spot"
  }'
```

**Response:**
```json
{
  "status": "completed",
  "success": true,
  "output": "4.0\n",
  "error": null,
  "exit_code": 0,
  "execution_time_seconds": 1.23,
  "tier_used": "cpu_basic",
  "job_id": "abc123..."
}
```

### Supported Languages

| Language | Image | File Extension |
|----------|-------|----------------|
| `python` | python:3.11-slim | .py |
| `node` / `javascript` | node:20-slim | .js |
| `go` | golang:1.22-alpine | .go |
| `rust` | rust:1.75-slim | .rs |

### Quota Management

Users have daily quotas for CPU and GPU time:

```bash
# Check remaining quota
curl http://localhost:8000/api/v1/validate/quota \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "cpu_seconds": 3200.5,
  "gpu_seconds": 540.0,
  "reset_in_hours": 18.5
}
```

| Quota Type | Default Daily Limit |
|------------|---------------------|
| CPU time | 3600 seconds (1 hour) |
| GPU time | 600 seconds (10 minutes) |

Quotas reset daily at midnight UTC.

### Security

Sandbox containers run with:
- **Network isolation**: No network access by default
- **Read-only filesystem**: Only `/tmp` is writable
- **Dropped capabilities**: All Linux capabilities dropped
- **Resource limits**: CPU, memory, and time limits enforced
- **No privilege escalation**: `--security-opt=no-new-privileges`

---

## Architecture Overview

![DOVA Architecture](./assets/dova-architecture.svg)

### System Components

<details>
<summary>Text-based Architecture Diagram</summary>

```
┌─────────────────────────────────────────────────────────────┐
│                      DOVA Platform                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   FastAPI   │    │    CLI      │    │   Workers   │     │
│  │   Server    │    │  Interface  │    │  (Async)    │     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            │                                │
│                   ┌────────▼────────┐                       │
│                   │  Orchestrator   │                       │
│                   │ (ReasoningMode) │                       │
│                   └────────┬────────┘                       │
│                            │                                │
│  ┌─────────────────────────┼─────────────────────────┐     │
│  │         COLLABORATIVE REASONING LAYER             │     │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  │     │
│  │  │ Blackboard │  │  Ensemble  │  │ Iterative  │  │     │
│  │  │  Service   │  │  Service   │  │ Refinement │  │     │
│  │  └────────────┘  └────────────┘  └────────────┘  │     │
│  │  ┌─────────────────────────────────────────────┐ │     │
│  │  │  Tool Resolver (Proactive Tool Discovery)   │ │     │
│  │  └─────────────────────────────────────────────┘ │     │
│  └─────────────────────────┬─────────────────────────┘     │
│                            │                                │
│         ┌──────────────────┼──────────────────┐             │
│         │                  │                  │             │
│  ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐     │
│  │  Research   │    │  Profiling  │    │ Validation  │     │
│  │   Agent     │    │   Agent     │    │   Agent     │     │
│  │ (ReAct+Mem) │    │ (ReAct+Mem) │    │ (ReAct+Mem) │     │
│  └──────┬──────┘    └─────────────┘    └─────────────┘     │
│         │                                                   │
│  ┌──────▼──────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Synthesis  │    │    Bull     │    │    Bear     │     │
│  │   Agent     │    │   Agent     │    │   Agent     │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                     Services Layer                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Source Registry                         │   │
│  │   (Built-in + Custom Sources, Quality Learning)      │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│    ┌────────────────────┼────────────────────┐             │
│    │                    │                    │             │
│  ┌─▼───────────┐  ┌─────▼─────┐  ┌──────────▼──────────┐  │
│  │  Built-in   │  │  Custom   │  │     Custom          │  │
│  │  MCP Tools  │  │  Web/RSS  │  │      APIs           │  │
│  └──────┬──────┘  └───────────┘  └─────────────────────┘  │
│         │                                                   │
│  ┌──────┴────────┬───────────────┬───────────────┐        │
│  │    ArXiv      │    GitHub     │  HuggingFace  │        │
│  └───────────────┴───────────────┴───────────────┘        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                   Infrastructure                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Amazon    │    │  AgentCore  │    │   Redis     │     │
│  │   Bedrock   │    │   Memory    │    │   Streams   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Background │    │   Sandbox   │    │  Proactive  │     │
│  │   Workers   │    │  (Docker)   │    │  Monitors   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

</details>

### Agent Responsibilities

| Agent | Purpose |
|-------|---------|
| **Orchestrator** | Intent classification, task routing, result aggregation, reasoning mode selection |
| **Research** | Search ArXiv, GitHub, HuggingFace via MCP + custom sources |
| **Profiling** | User preference tracking, temporal interest decay |
| **Validation** | Code quality, security analysis, best practices |
| **Synthesis** | Cross-source result synthesis, insight generation |
| **Bull/Bear** | Balanced technology evaluation through debate |

### Reasoning Services

| Service | Purpose |
|---------|---------|
| **ReasoningMixin** | ReAct loops, self-reflection, working memory for all agents |
| **Blackboard** | Shared workspace for posting hypotheses, evidence, and votes |
| **Ensemble** | Parallel reasoning with aggregation (synthesis, vote, best-of, union) |
| **Collaborative** | Unified orchestrator combining blackboard, ensemble, iterative, and tool-augmented |
| **ToolResolver** | Proactive tool discovery and execution plan creation |
| **TaskAnalyzer** | Pattern-based task analysis to detect required capabilities |

### Services

| Service | Purpose |
|---------|---------|
| **Source Registry** | Manage built-in and custom sources per user |
| **Source Fetcher** | Fetch content from Web URLs, RSS feeds, and APIs |
| **Quality Tracker** | Learn source quality from implicit user signals |
| **Blackboard** | Shared workspace for multi-agent collaborative reasoning |
| **Ensemble Reasoning** | Coordinate parallel agent reasoning with result aggregation |
| **Collaborative Reasoning** | Orchestrate hybrid reasoning patterns (blackboard + ensemble + iterative) |
| **Job Queue** | Redis Streams-based reliable job queue with consumer groups |
| **Job Scheduler** | APScheduler-based periodic job scheduling (ArXiv/HF polling) |
| **Content Processor** | Normalize and embed content for recommendation matching |
| **User Matcher** | Match content to users based on similarity and subscriptions |
| **Delivery Manager** | Batch recommendations with frequency capping |
| **Subscription Manager** | CRUD operations for user content subscriptions |
| **Sandbox Scheduler** | Infer resource tier from code analysis |
| **Sandbox Executor** | Docker-based isolated code execution |
| **Quota Manager** | Track and enforce user execution quotas |

### Request Flow

1. **User Request** → API receives research query with optional reasoning mode
2. **Intent Classification** → Orchestrator determines intent type
3. **Reasoning Mode Selection** → Choose quick/standard/deep/collaborative/tool_augmented based on request
4. **Tool Discovery** (tool_augmented mode) → Analyze task, discover tools, create execution plan
5. **Tool Execution** (tool_augmented mode) → Execute selected tools in priority order
6. **Task Graph** → Build dependency graph of subtasks
7. **Agent Reasoning** → Each agent uses ReAct loop with reflection (if standard+)
8. **Collaborative Reasoning** → For deep/collaborative modes, run ensemble or blackboard patterns
9. **Synthesis** → Combine results into coherent response with confidence scores
10. **Profiling** → Update user preferences based on interaction

---

## Deployment

### Deploy to AWS

```bash
# Set environment
export ENVIRONMENT=staging  # or prod
export AWS_REGION=us-east-1

# Deploy infrastructure
./scripts/deploy.sh --environment staging

# For production (requires confirmation)
./scripts/deploy.sh --environment prod
```

### Deployment Components

The CDK stack deploys:

- **API Gateway** - REST API endpoint
- **Lambda** - Serverless compute for API
- **DynamoDB** - User profiles, session storage
- **S3** - Research artifacts, cache
- **ElastiCache** - Redis for caching
- **Cognito** - User authentication
- **CloudWatch** - Logging and monitoring

### Environment-Specific Configuration

| Environment | Features |
|-------------|----------|
| **dev** | Debug logging, API docs enabled, no auth required |
| **staging** | Production-like, with test data |
| **prod** | Full security, monitoring, auto-scaling |

### Post-Deployment Verification

```bash
# Get API URL from CDK outputs
API_URL=$(cat infra/cdk-outputs.json | jq -r '.["dova-staging"].ApiUrl')

# Test health endpoint
curl $API_URL/health

# Test research endpoint
curl -X POST $API_URL/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{"query": "test query"}'
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

```
ModuleNotFoundError: No module named 'dova'
```

**Solution:** Ensure you've installed in development mode:
```bash
pip install -e ".[dev]"
```

#### 2. AWS Credentials

```
botocore.exceptions.NoCredentialsError
```

**Solution:** Configure AWS credentials:
```bash
aws configure
# Or set environment variables
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

#### 3. Bedrock Access Denied

```
AccessDeniedException: You don't have access to the model
```

**Solution:**
1. Go to AWS Console → Bedrock → Model access
2. Request access to Claude models
3. Wait for approval (usually instant)

#### 4. MCP Server Connection Failures

```
MCPConnectionError: Failed to connect to arxiv server
```

**Solution:**

1. Check if MCP servers are configured:
   ```bash
   dova mcp list
   ```

2. If no servers configured, add them:
   ```bash
   dova mcp add arxiv --url http://infs.cavatar.info:8084/mcp
   dova mcp add huggingface --url https://huggingface.co/mcp
   dova mcp add github --url https://api.githubcopilot.com/mcp -H ghp_yourtoken
   ```

3. Test server connectivity:
   ```bash
   dova mcp test arxiv
   dova mcp test huggingface --tool model_search
   ```

4. Check `~/.dova.json` for correct configuration:
   ```bash
   dova mcp show
   ```

5. Verify server names match expected names: `arxiv`, `github`, `huggingface`

#### 5. Redis Connection Failed

```
redis.exceptions.ConnectionError
```

**Solution:**
```bash
# Check Redis status (if installed as system service)
sudo systemctl status redis

# Start Redis with Docker
docker-compose up -d redis

# Or disable Redis caching
export REDIS_ENABLED=false
```

#### 6. Sandbox Execution Disabled

```
HTTPException: Sandbox execution not enabled
```

**Solution:**
```bash
# Enable sandbox in environment
export SANDBOX_ENABLED=true

# Ensure Docker is running and accessible
docker info

# For Docker-in-Docker setups, mount the socket
# -v /var/run/docker.sock:/var/run/docker.sock
```

#### 7. Quota Exceeded

```
HTTPException: Quota exceeded: CPU quota exceeded. Remaining: 0s
```

**Solution:**
- Wait for quota reset (midnight UTC)
- Or increase default quotas in settings:
  ```bash
  export SANDBOX_DEFAULT_CPU_QUOTA_SECONDS=7200
  export SANDBOX_DEFAULT_GPU_QUOTA_SECONDS=1200
  ```

#### 8. Background Workers Not Processing

```
Jobs stuck in pending state
```

**Solution:**
```bash
# Start workers
docker-compose up -d worker

# Or run worker manually
python -m dova.jobs.worker

# Check Redis stream
redis-cli XLEN dova:jobs
redis-cli XINFO GROUPS dova:jobs
```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
# Via environment
export LOG_LEVEL=DEBUG

# Via CLI
dova --debug serve

# In Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Getting Help

- **Documentation**: See `/docs` folder
- **Issues**: Report bugs on GitHub
- **Logs**: Check CloudWatch in production

---

## Next Steps

1. **Explore the API** - Try different research queries
2. **Experiment with Reasoning Modes** - Try `quick`, `standard`, `deep`, `collaborative`, and `tool_augmented` modes
3. **Add Custom Sources** - Configure RSS feeds, web scrapers, or custom APIs
4. **Set Up Subscriptions** - Subscribe to ArXiv categories, HuggingFace tasks, or GitHub topics
5. **Enable Sandbox Execution** - Set `SANDBOX_ENABLED=true` for code execution
6. **Start Background Workers** - Run `docker-compose up worker` for proactive monitoring
7. **Customize Agents** - Extend agents with custom actions in the ReAct loop
8. **Implement Custom Collaboration** - Use blackboard/ensemble/tool-augmented patterns for your domain
9. **Use Tool-Augmented Mode** - Let DOVA automatically discover and use the best tools for your tasks
10. **Deploy to Production** - Use the CDK stack

For more detailed documentation, see:
- [Architecture Guide](./architecture.md)
- [API Reference](./api-reference.md)
- [Deployment Guide](./deployment.md)
