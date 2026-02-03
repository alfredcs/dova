# DOVA Getting Started Guide

This guide will walk you through setting up, configuring, and using the DOVA (Deep Orchestrated Versatile Agent Platform) for AI/ML research automation.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Automated AWS Setup](#automated-aws-setup-agentcore)
5. [Running Locally](#running-locally)
6. [Browser-Based Research UI](#browser-based-research-ui) *(New in v1.3)*
7. [Using the CLI](#using-the-cli)
8. [API Usage](#api-usage)
9. [Deep Research Features](#deep-research-features) *(New in v1.3)*
10. [Agentic Reasoning](#agentic-reasoning)
11. [Advanced Agent Intelligence](#advanced-agent-intelligence)
12. [Managing Custom Sources](#managing-custom-sources)
13. [Proactive Recommendations](#proactive-recommendations)
14. [Sandbox Execution](#sandbox-execution)
15. [Architecture Overview](#architecture-overview)
16. [Deployment](#deployment)
17. [Troubleshooting](#troubleshooting)

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

# Web Search Providers (optional - DuckDuckGo is free fallback)
# Provider priority: Brave > Perplexity > Tavily > DuckDuckGo
BRAVE_API_KEY=               # https://brave.com/search/api/
PERPLEXITY_API_KEY=          # https://perplexity.ai/settings/api
TAVILY_API_KEY=              # https://tavily.com

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

# Advanced Intelligence (OpenClaw-inspired)
THINKING_DEFAULT_LEVEL=medium
THINKING_AUTO_SELECT_ENABLED=true
EVAL_AUTO_EVALUATE_RESPONSES=false
EVAL_MIN_CONFIDENCE_THRESHOLD=0.6
SESSION_STALE_AFTER_SECONDS=1800
SESSION_EXPIRE_AFTER_SECONDS=86400
DISCOVERY_AUTO_DISCOVER_ON_STARTUP=true
MEMORY_ENHANCED_MMR_LAMBDA=0.5
HEARTBEAT_ENABLED=true
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

### Automated AWS Setup (AgentCore)

DOVA provides automated setup for all AWS services required for AgentCore deployment. This creates IAM roles, Cognito resources, SSM parameters, and Secrets Manager secrets.

#### Prerequisites

1. **AWS CLI configured** with credentials that have sufficient permissions
2. **View required permissions**:
   ```bash
   dova aws permissions
   ```

#### Quick Setup

```bash
# Run automated setup with a custom stack name
dova aws setup --stack-name my-dova-stack --region us-east-1

# Or with additional options
dova aws setup \
  --stack-name my-dova-stack \
  --region us-east-1 \
  --gateway-url https://your-gateway.bedrock.us-east-1.amazonaws.com \
  --memory-id your-memory-id
```

#### What Gets Created

| Service | Resources Created |
|---------|-------------------|
| **IAM** | Execution role + 4 policies (Bedrock, AgentCore, SSM, Secrets) |
| **Cognito** | User Pool, Resource Server, App Client, Domain |
| **SSM Parameter Store** | `/stack-name/cognito_provider`, `/stack-name/machine_client_id` |
| **Secrets Manager** | `/stack-name/machine_client_secret` |

#### Validation

After setup, validate your configuration:

```bash
# Validate all AWS resources
dova aws validate --stack-name my-dova-stack

# Generate environment file
dova aws env --stack-name my-dova-stack -o .env.aws

# Source the environment file
source .env.aws
```

#### Cleanup

To remove all created resources:

```bash
dova aws teardown --stack-name my-dova-stack
```

#### AWS CLI Commands

| Command | Description |
|---------|-------------|
| `dova aws setup` | Create all AWS resources for DOVA |
| `dova aws validate` | Validate existing AWS setup |
| `dova aws teardown` | Remove all DOVA AWS resources |
| `dova aws permissions` | Show required IAM permissions |
| `dova aws env` | Generate environment file from existing setup |

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

### Web Search Configuration

DOVA supports multi-provider web search with automatic selection and fallback. **Web search works out of the box** using DuckDuckGo as a free fallback - no API key required.

**Provider Priority (auto mode):**
1. **Brave Search** - When `BRAVE_API_KEY` is set (structured results, freshness filtering)
2. **Perplexity Sonar** - When `PERPLEXITY_API_KEY` is set (AI-synthesized answers)
3. **Tavily** - When `TAVILY_API_KEY` is set (advanced search)
4. **DuckDuckGo** - Always available (free, no setup needed)

**Configuration (optional - for better results):**
```bash
# Configure one or more providers for improved search quality
export BRAVE_API_KEY=xxx           # https://brave.com/search/api/
export PERPLEXITY_API_KEY=xxx      # https://perplexity.ai/settings/api
export TAVILY_API_KEY=tvly-xxx     # https://tavily.com

# Or use the prefixed version for Tavily
export MCP_TAVILY_API_KEY=tvly-xxxxxxxxxxxx
```

**Quick Test:**
```bash
# Works immediately - no API keys needed
dova research "latest AI news" -s web
```

The orchestrator intelligently selects web search for news-related queries (e.g., queries containing "latest", "news", "nominated", etc.).

### Model Tiering Configuration

DOVA uses a tiered model selection system to optimize costs. Simpler tasks use faster/cheaper models, while complex tasks use more capable models.

| Tier | Task Types | Default Bedrock Model |
|------|------------|----------------------|
| `basic` | Classification, summarization | claude-haiku-4-5 |
| `standard` | General queries | claude-sonnet-4 |
| `advanced` | Code generation, research | claude-sonnet-4 |
| `reasoning` | Complex reasoning, synthesis | claude-sonnet-4 |

**Override with environment variables:**

```bash
# Custom models per tier
export LLM_TIER_BASIC_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0
export LLM_TIER_STANDARD_MODEL=us.anthropic.claude-sonnet-4-20250514-v1:0
export LLM_TIER_ADVANCED_MODEL=us.anthropic.claude-sonnet-4-20250514-v1:0
export LLM_TIER_REASONING_MODEL=us.anthropic.claude-sonnet-4-20250514-v1:0
```

**View current configuration:**

```bash
dova models
```

---

## Running Locally

### Start the API Server

```bash
# Using the CLI (FastAPI mode - default)
dova serve

# Or with uvicorn directly
uvicorn dova.api.main:app --reload --host 0.0.0.0 --port 8000

# Or using Make
make run-local
```

### AgentCore Runtime Mode

For AWS deployment with AgentCore Runtime:

```bash
# Set up AWS resources first (if not done)
dova aws setup --stack-name my-dova-stack

# Start in AgentCore mode
dova serve --mode agentcore
```

This mode uses the BedrockAgentCoreApp runtime with:
- OAuth2 authentication via Cognito
- AgentCore Memory for persistent context
- AgentCore Gateway for MCP tools

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

## Browser-Based Research UI

*(New in v1.3)*

DOVA now includes a browser-based research interface for interactive queries.

### Starting the UI

```bash
# Start the server
dova serve --port 8081

# Open in browser
open http://localhost:8081
```

### UI Features

The research UI provides:

1. **Query Input**: Enter natural language research questions
2. **Source Selection**: Toggle chips for:
   - ArXiv Papers
   - GitHub Repositories
   - HuggingFace Models
   - Web Search

3. **Research Answer**: Direct synthesized answer with confidence score
   - High confidence (≥70%): Green badge
   - Medium confidence (40-70%): Yellow badge
   - Low confidence (<40%): Red badge

4. **Organized Results**: Results grouped by source type:
   - Papers with authors and publication date
   - Repositories with stars and language
   - Models with downloads and pipeline type
   - Web results with descriptions

5. **Query Refinement**: Displays "Query refined N times" when DOVA improved the search

### Example Queries

```text
# Technical query (searches ArXiv, GitHub, HuggingFace, Web)
"transformer attention mechanism implementation"

# Biographical query (searches Web only - smart routing)
"explain elon musk's college education background"

# Code-focused query (prioritizes GitHub)
"most starred repo on github with agentic reasoning"
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

# Include web search for current events (works out of the box via DuckDuckGo)
dova research "latest AI breakthroughs" -s arxiv -s web

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

### Model Configuration

```bash
# Show model tiering configuration
dova models

# Output shows:
# - Current provider and default model
# - Model tiers (Basic, Standard, Advanced, Reasoning)
# - Task-to-tier mappings
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

### MCP Server Repositories

DOVA can automatically clone and manage MCP server repositories (like `arxiv-mcp-server`). These are cloned to `~/.dova/mcp-servers/`.

```bash
# Setup MCP server repos (clone & install dependencies)
dova mcp setup

# Setup a specific repo
dova mcp setup --name arxiv

# Update repos (git pull)
dova mcp update

# List managed repos
dova mcp repos
```

**Automatic Weekly Updates**: When running `dova serve`, a heartbeat task automatically git pulls MCP server repos every Sunday at 4 AM.

**Default Managed Repos:**

| Name | Repository | Description |
|------|------------|-------------|
| `arxiv` | [blazickjp/arxiv-mcp-server](https://github.com/blazickjp/arxiv-mcp-server) | ArXiv paper search and download |

---

## API Usage

### Authentication

DOVA supports two authentication methods:

1. **JWT Token** (for user applications) - Issued by AWS Cognito
2. **API Key** (for programmatic access) - Generated via DOVA API

#### Configuring AWS Cognito for JWT Authentication

JWT tokens are issued by AWS Cognito. To enable JWT authentication:

**1. Set Environment Variables**

```bash
# Required Cognito configuration
export AUTH_COGNITO_USER_POOL_ID=us-east-1_XXXXXXXXX
export AUTH_COGNITO_CLIENT_ID=your-cognito-app-client-id

# Optional: KMS key for credential encryption
export AUTH_KMS_KEY_ID=alias/dova-credentials
```

**2. Create a Cognito User Pool (if not using CDK deployment)**

```bash
# Create user pool
aws cognito-idp create-user-pool \
  --pool-name dova-users \
  --policies 'PasswordPolicy={MinimumLength=8,RequireUppercase=true,RequireLowercase=true,RequireNumbers=true}' \
  --auto-verified-attributes email \
  --schema Name=email,Required=true,Mutable=true

# Note the UserPoolId from the response

# Create app client (no secret for public clients)
aws cognito-idp create-user-pool-client \
  --user-pool-id <UserPoolId> \
  --client-name dova-api \
  --no-generate-secret \
  --explicit-auth-flows ALLOW_USER_PASSWORD_AUTH ALLOW_REFRESH_TOKEN_AUTH

# Note the ClientId from the response
```

**3. User Registration and Login**

```bash
# Sign up a new user
aws cognito-idp sign-up \
  --client-id <ClientId> \
  --username user@example.com \
  --password "YourSecurePassword123!" \
  --user-attributes Name=email,Value=user@example.com

# Confirm the user (admin action, or user confirms via email)
aws cognito-idp admin-confirm-sign-up \
  --user-pool-id <UserPoolId> \
  --username user@example.com

# Sign in to get JWT tokens
aws cognito-idp initiate-auth \
  --client-id <ClientId> \
  --auth-flow USER_PASSWORD_AUTH \
  --auth-parameters USERNAME=user@example.com,PASSWORD="YourSecurePassword123!"
```

The response contains:
- `IdToken` - Use this as the Bearer token for DOVA API
- `AccessToken` - Can also be used for authentication
- `RefreshToken` - Use to obtain new tokens when they expire

**4. Using the JWT Token**

```bash
curl -H "Authorization: Bearer <IdToken>" \
     http://localhost:8000/api/v1/research
```

#### Obtaining an API Key

API keys provide programmatic access without going through the Cognito login flow. You must first authenticate (via JWT or in development mode) to create an API key.

**1. Create an API Key (requires authentication)**

```bash
# Using JWT authentication
curl -X POST http://localhost:8000/api/v1/credentials/api-keys \
  -H "Authorization: Bearer <jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-cli-key",
    "roles": ["api_user"],
    "expires_in_days": 365
  }'
```

**Response:**
```json
{
  "key_id": "dova_key_a1b2c3d4",
  "api_key": "dova_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456789",
  "name": "my-cli-key",
  "roles": ["api_user"],
  "expires_at": "2027-01-31T00:00:00Z"
}
```

> **Important:** The `api_key` value is shown only once. Store it securely.

**2. Using the API Key**

```bash
curl -H "X-API-Key: dova_aBcDeFgHiJkLmNoPqRsTuVwXyZ123456789" \
     http://localhost:8000/api/v1/research
```

**3. Managing API Keys**

```bash
# List your API keys (metadata only, secrets not shown)
curl http://localhost:8000/api/v1/credentials/api-keys \
  -H "Authorization: Bearer <jwt-token>"

# Revoke an API key
curl -X DELETE http://localhost:8000/api/v1/credentials/api-keys/dova_key_a1b2c3d4 \
  -H "Authorization: Bearer <jwt-token>"
```

#### Development Mode (No Auth Required)

In development mode (`ENVIRONMENT=development`), authentication is optional:

```bash
# Anonymous access (development only)
curl http://localhost:8000/api/v1/research

# Or use any key starting with "dova_" (development only)
curl -H "X-API-Key: dova_test123" \
     http://localhost:8000/api/v1/research
```

> **Warning:** Development mode bypasses real authentication. Never use in production.

#### Authentication Summary

| Method | Source | Use Case | Production |
|--------|--------|----------|------------|
| JWT Token | AWS Cognito | User apps with login flow | ✅ Yes |
| API Key | DOVA `/credentials/api-keys` | CLI tools, scripts, automation | ✅ Yes |
| Anonymous | N/A | Local development | ❌ Dev only |
| `dova_*` prefix | N/A | Quick testing | ❌ Dev only |

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
  -H "Content-Type: application/json" \
  -d '{"query": "pytorch transformers", "max_results": 10}'

# Search HuggingFace models
curl -X POST http://localhost:8000/api/v1/search/huggingface \
  -H "Content-Type: application/json" \
  -d '{"query": "text generation", "max_results": 10}'
```

### Profile Management

```bash
# Get user profile
curl http://localhost:8000/api/v1/profile \
  -H "Authorization: Bearer <token>"

# Update profile
curl -X PUT http://localhost:8000/api/v1/profile \
  -H "Content-Type: application/json" \
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
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "code": "def hello(): print(\"world\")",
    "language": "python"
  }'

# Execute code in sandbox (requires SANDBOX_ENABLED=true)
curl -X POST http://localhost:8000/api/v1/validate/execute \
  -H "Content-Type: application/json" \
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

## Deep Research Features

*(New in v1.3)*

DOVA v1.3 transforms from a link aggregator into a true deep research assistant with intelligent query understanding, answer synthesis, and iterative refinement.

### Query Type Classification

DOVA automatically classifies queries into types for smart source routing:

| Query Type | Example | Sources Searched |
|------------|---------|------------------|
| `technical` | "transformer attention mechanism" | ArXiv, GitHub, HuggingFace, Web |
| `biographical` | "elon musk's education" | Web only |
| `factual` | "what is machine learning" | Web primarily |
| `general` | "AI trends 2026" | All sources |

### Answer Synthesis

Instead of just returning links, DOVA synthesizes a direct answer:

```json
{
  "query": "explain transformer architecture",
  "answer": "The Transformer architecture, introduced in 'Attention Is All You Need'...",
  "confidence": 0.85,
  "refinement_attempts": 0,
  "papers": [...],
  "repositories": [...]
}
```

### Confidence Scoring

Each answer receives a confidence score (0.0-1.0):

- **High (≥0.7)**: Answer is comprehensive and well-supported
- **Medium (0.4-0.7)**: Answer is partial, may need verification
- **Low (<0.4)**: Answer is tentative, recommend further research

### Iterative Query Refinement

When confidence is below threshold (70%), DOVA automatically:

1. Analyzes what information is missing
2. Generates a refined search query
3. Re-executes research with improved query
4. Repeats up to 2 times

Example:
```text
Original: "latest AI developments"
Refined:  "recent advances in large language models and generative AI 2026"
```

### Memory-Assisted Research

Research results are stored in memory for future reference:

- **Short-Term Memory**: All research stored with 24-hour TTL
- **Long-Term Memory**: High-confidence (≥70%) answers stored persistently
- **Semantic Search**: Embeddings enable finding similar past research

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

## Advanced Agent Intelligence

DOVA includes OpenClaw-inspired features for enhanced agent intelligence, including adaptive thinking, self-evaluation, session management, and proactive maintenance.

### Multi-Tiered Thinking System

The thinking system provides configurable reasoning depth based on task complexity:

| Level | Budget Tokens | Use Case |
|-------|---------------|----------|
| `off` | 0 | Simple lookups, no reasoning needed |
| `minimal` | 1,024 | Quick classifications, simple decisions |
| `low` | 4,096 | Summarization, basic analysis |
| `medium` | 16,384 | Code generation, moderate reasoning (default) |
| `high` | 32,768 | Complex research, multi-step analysis |
| `xhigh` | 65,536 | Deep reasoning, comprehensive research |

```python
from dova.services.thinking import ThinkingService, ThinkingLevel

# Initialize service
thinking = ThinkingService(default_level=ThinkingLevel.MEDIUM)

# Auto-select level based on task
level = thinking.select_level_for_task(
    task_type="reasoning",
    query="Compare transformer architectures for sequence modeling",
    complexity_hint="complex"
)

# Create thinking parameters for LLM request
params = thinking.create_thinking_params(level)
# Returns: {"thinking": {"type": "enabled", "budget_tokens": 32768}}
```

**Configuration:**
```bash
THINKING_DEFAULT_LEVEL=medium
THINKING_AUTO_SELECT_ENABLED=true
THINKING_MAX_BUDGET_TOKENS=65536
```

### Self-Evaluation and Error Diagnosis

The evaluation service assesses response quality and diagnoses errors:

```python
from dova.services.evaluation import SelfEvaluator, ErrorType

evaluator = SelfEvaluator(min_confidence=0.6)

# Evaluate a response
result = await evaluator.evaluate(
    response="The transformer architecture uses self-attention...",
    prompt="Explain transformer architecture",
    expected_format="markdown"
)

print(f"Confidence: {result.confidence}")  # 0.0 to 1.0
print(f"Should retry: {result.should_retry}")
print(f"Caveats: {result.caveats}")

# Diagnose an error
diagnosis = evaluator.diagnose_error(
    error="Rate limit exceeded",
    context={"provider": "anthropic", "model": "claude-3"}
)

print(f"Error type: {diagnosis.error_type}")  # TRANSIENT
print(f"Action: {diagnosis.action}")  # RETRY_WITH_BACKOFF
print(f"Retry after: {diagnosis.retry_after_seconds}s")
```

**Error Types and Recovery Actions:**

| Error Type | Description | Default Action |
|------------|-------------|----------------|
| `transient` | Temporary failures (rate limits, timeouts) | Retry with backoff |
| `configuration` | Setup issues (invalid keys, permissions) | Alert user |
| `capability` | Model limitations (context length, unsupported) | Fallback to different model |
| `unknown` | Unclassified errors | Log and continue |

**Configuration:**
```bash
EVAL_AUTO_EVALUATE_RESPONSES=false
EVAL_MIN_CONFIDENCE_THRESHOLD=0.6
```

### Session Management

The session service tracks user sessions with automatic freshness evaluation:

```python
from dova.services.session import SessionManager, SessionAction
from dova.utils.cache import InMemoryCache

cache = InMemoryCache()
session_mgr = SessionManager(
    cache=cache,
    stale_after_seconds=1800,   # 30 minutes
    expire_after_seconds=86400  # 24 hours
)

# Create a session
session = await session_mgr.create_session(
    user_id="user-123",
    context={"query_history": [], "preferences": {}}
)

# Update session activity
session = await session_mgr.update_activity(session.id)

# Evaluate session freshness
state, action = session_mgr.evaluate_freshness(session)

# Execute recovery action if needed
if action != SessionAction.CONTINUE:
    session = await session_mgr.execute_action(session, action)
```

**Session States and Actions:**

| State | Trigger | Recommended Action |
|-------|---------|-------------------|
| `active` | Recent activity | Continue normally |
| `stale` | No activity for 30+ min | Refresh context |
| `expired` | No activity for 24+ hours | Fork to new session |

**Configuration:**
```bash
SESSION_STALE_AFTER_SECONDS=1800
SESSION_EXPIRE_AFTER_SECONDS=86400
```

### Enhanced Memory with Semantic Search

The enhanced memory service provides semantic search with MMR diversity ranking:

```python
from dova.services.memory_enhanced import EnhancedMemoryService, MemoryType
from dova.utils.cache import InMemoryCache

cache = InMemoryCache()
memory = EnhancedMemoryService(
    cache=cache,
    llm_router=llm_router,  # For embeddings
    mmr_lambda=0.5  # Balance relevance (1.0) vs diversity (0.0)
)

# Store a memory
entry_id = await memory.store(
    memory_type=MemoryType.LONG_TERM,
    content={"text": "User prefers detailed technical explanations"},
    importance=0.8,
    user_id="user-123",
    tags=["preference", "technical"]
)

# Semantic search with MMR reranking
results = await memory.search_semantic(
    query="technical documentation style",
    user_id="user-123",
    top_k=5,
    use_mmr=True  # Ensures diverse results
)

for result in results:
    print(f"Score: {result.score}, Content: {result.entry.content}")
```

**Memory Types:**

| Type | Description | Persistence |
|------|-------------|-------------|
| `short_term` | Session-scoped, temporary | 24 hours TTL |
| `long_term` | Persistent across sessions | No expiry |
| `procedural` | How-to knowledge, skills | No expiry |

**Configuration:**
```bash
MEMORY_ENHANCED_SEMANTIC_SEARCH_ENABLED=true
MEMORY_ENHANCED_MMR_LAMBDA=0.5
MEMORY_ENHANCED_EMBEDDING_CACHE_TTL=3600
```

### Auto-Discovery Service

The discovery service automatically finds available models and MCP servers:

```python
from dova.services.discovery import AutoDiscovery
from dova.utils.cache import InMemoryCache

cache = InMemoryCache()
discovery = AutoDiscovery(
    cache=cache,
    llm_router=llm_router,
    cache_ttl=3600
)

# Discover available models
models = await discovery.discover_models()
for model in models:
    print(f"{model.provider}/{model.model_id}: {model.capabilities}")

# Find a model with specific capability
vision_model = await discovery.get_model_by_capability(
    capability="vision",
    prefer_provider="bedrock"
)

# Discover MCP servers
servers = await discovery.discover_mcp_servers()
for server in servers:
    print(f"{server.name}: {server.tools}, healthy={server.healthy}")

# Refresh all discovery caches
summary = await discovery.refresh_all()
print(f"Found {summary['models']['total']} models, {summary['mcp_servers']['total']} MCP servers")
```

**Configuration:**
```bash
DISCOVERY_AUTO_DISCOVER_ON_STARTUP=true
DISCOVERY_CACHE_TTL_SECONDS=3600
```

### Proactive Heartbeat System

The heartbeat processor runs scheduled maintenance tasks:

```python
from dova.jobs.heartbeat import HeartbeatProcessor, HeartbeatTask

# Initialize with default tasks
heartbeat = HeartbeatProcessor(auto_register_defaults=True)

# Default tasks include:
# - subscription_monitor: */15 * * * * (every 15 min)
# - recommendation_refresh: 0 */4 * * * (every 4 hours)
# - mcp_health_check: */5 * * * * (every 5 min)
# - session_cleanup: 0 3 * * * (daily at 3 AM)

# Add a custom task
heartbeat.register_task(HeartbeatTask(
    name="cache_warmup",
    cron_schedule="0 */6 * * *",  # Every 6 hours
    handler="warmup_caches",
))

# Register a custom handler
async def warmup_caches():
    print("Warming up caches...")

heartbeat.register_handler("warmup_caches", warmup_caches)

# Start the heartbeat processor
await heartbeat.start()

# Run a task immediately
await heartbeat.run_task_now("mcp_health_check")

# Stop gracefully
await heartbeat.stop()
```

**Configuration:**
```bash
HEARTBEAT_ENABLED=true
HEARTBEAT_SUBSCRIPTION_MONITOR_CRON="*/15 * * * *"
HEARTBEAT_RECOMMENDATION_REFRESH_CRON="0 */4 * * *"
HEARTBEAT_MCP_HEALTH_CHECK_CRON="*/5 * * * *"
HEARTBEAT_SESSION_CLEANUP_CRON="0 3 * * *"
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
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"type": "arxiv_keyword", "value": "transformer"}'

# Subscribe to HuggingFace task
curl -X POST http://localhost:8000/api/v1/subscriptions \
  -H "Content-Type: application/json" \
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
  -H "Content-Type: application/json" \
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
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "code": "import numpy as np\nprint(np.array([1,2,3]).sum())",
    "language": "python",
    "dependencies": ["numpy"]
  }'

# Specify tier explicitly
curl -X POST http://localhost:8000/api/v1/validate/execute \
  -H "Content-Type: application/json" \
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

### Advanced Intelligence Services

| Service | Purpose |
|---------|---------|
| **ThinkingService** | Multi-tiered thinking budgets with auto-selection based on task complexity |
| **SelfEvaluator** | Response quality assessment, confidence scoring, and error diagnosis |
| **SessionManager** | Session lifecycle management with freshness evaluation and recovery |
| **EnhancedMemoryService** | Semantic search with embeddings and MMR diversity reranking |
| **AutoDiscovery** | Runtime discovery of models and MCP servers with capability caching |
| **HeartbeatProcessor** | Cron-based proactive maintenance tasks |

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

### Quick Deployment with Automated AWS Setup

The easiest way to deploy DOVA is using the automated AWS setup:

```bash
# 1. Set up AWS resources (IAM, Cognito, SSM, Secrets)
dova aws setup --stack-name my-dova-stack --region us-east-1

# 2. Source the generated environment file
source .env.aws

# 3. Start DOVA in AgentCore mode
dova serve --mode agentcore
```

### Deploy to AWS (Full Infrastructure)

For a complete infrastructure deployment with CDK:

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

#### 3b. AWS Setup Permission Errors

```
IAM setup failed: Access denied
```

**Solution:**
1. Check required permissions:
   ```bash
   dova aws permissions
   ```
2. Ensure your AWS user/role has the listed IAM permissions
3. Attach a policy with the required actions to your user/role

Common missing permissions:
- `iam:CreateRole`, `iam:CreatePolicy`, `iam:AttachRolePolicy`
- `cognito-idp:CreateUserPool`, `cognito-idp:CreateUserPoolClient`
- `ssm:PutParameter`, `secretsmanager:CreateSecret`

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
3. **Configure Thinking Levels** - Adjust `THINKING_DEFAULT_LEVEL` for your use case
4. **Enable Self-Evaluation** - Set `EVAL_AUTO_EVALUATE_RESPONSES=true` for quality monitoring
5. **Add Custom Sources** - Configure RSS feeds, web scrapers, or custom APIs
6. **Set Up Subscriptions** - Subscribe to ArXiv categories, HuggingFace tasks, or GitHub topics
7. **Enable Sandbox Execution** - Set `SANDBOX_ENABLED=true` for code execution
8. **Start Background Workers** - Run `docker-compose up worker` for proactive monitoring
9. **Enable Heartbeat Tasks** - Configure cron schedules for automated maintenance
10. **Use Enhanced Memory** - Leverage semantic search for better recall
11. **Customize Agents** - Extend agents with custom actions in the ReAct loop
12. **Implement Custom Collaboration** - Use blackboard/ensemble/tool-augmented patterns for your domain
13. **Use Tool-Augmented Mode** - Let DOVA automatically discover and use the best tools for your tasks
14. **Deploy to Production** - Use the CDK stack

For more detailed documentation, see:
- [Architecture Guide](./architecture.md)
- [API Reference](./api-reference.md)
- [Deployment Guide](./deployment.md)
