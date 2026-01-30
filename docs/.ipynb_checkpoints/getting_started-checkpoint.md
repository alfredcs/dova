# DOVA Getting Started Guide

This guide will walk you through setting up, configuring, and using the DOVA (Deep Orchestrated Versatile Agent Platform) for AI/ML research automation.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running Locally](#running-locally)
5. [Using the CLI](#using-the-cli)
6. [API Usage](#api-usage)
7. [Architecture Overview](#architecture-overview)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)

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

# Redis (optional, for caching)
REDIS_URL=redis://localhost:6379
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

DOVA integrates with Model Context Protocol (MCP) servers for external data sources:

| Server | Purpose | Configuration |
|--------|---------|---------------|
| ArXiv | Academic papers | No API key required |
| GitHub | Code repositories | `GITHUB_TOKEN` (optional, increases rate limits) |
| HuggingFace | ML models/datasets | `HF_TOKEN` (optional) |

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

# View logs
docker-compose logs -f dova-api

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
# Validate code snippet
curl -X POST http://localhost:8000/api/v1/validate/code \
  -H "Authorization: Bearer <token>" \
  -d '{
    "code": "def hello(): print(\"world\")",
    "language": "python"
  }'

# Validate repository
curl -X POST http://localhost:8000/api/v1/validate/repository \
  -d '{"repository_url": "https://github.com/user/repo"}'
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

## Architecture Overview

### System Components

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
│                   │     Agent       │                       │
│                   └────────┬────────┘                       │
│                            │                                │
│         ┌──────────────────┼──────────────────┐             │
│         │                  │                  │             │
│  ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐     │
│  │  Research   │    │  Profiling  │    │ Validation  │     │
│  │   Agent     │    │   Agent     │    │   Agent     │     │
│  └──────┬──────┘    └─────────────┘    └─────────────┘     │
│         │                                                   │
│  ┌──────▼──────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Synthesis  │    │    Bull     │    │    Bear     │     │
│  │   Agent     │    │   Agent     │    │   Agent     │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                     External Services                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Amazon    │    │    MCP      │    │   Redis     │     │
│  │   Bedrock   │    │  Servers    │    │   Cache     │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                            │                                │
│         ┌──────────────────┼──────────────────┐             │
│         │                  │                  │             │
│  ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐     │
│  │   ArXiv     │    │   GitHub    │    │ HuggingFace │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Agent Responsibilities

| Agent | Purpose |
|-------|---------|
| **Orchestrator** | Intent classification, task routing, result aggregation |
| **Research** | Search ArXiv, GitHub, HuggingFace via MCP |
| **Profiling** | User preference tracking, temporal interest decay |
| **Validation** | Code quality, security analysis, best practices |
| **Synthesis** | Cross-source result synthesis, insight generation |
| **Bull/Bear** | Balanced technology evaluation through debate |

### Request Flow

1. **User Request** → API receives research query
2. **Intent Classification** → Orchestrator determines intent type
3. **Task Graph** → Build dependency graph of subtasks
4. **Parallel Execution** → Execute independent tasks concurrently
5. **Synthesis** → Combine results into coherent response
6. **Profiling** → Update user preferences based on interaction

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
- Check network connectivity
- Verify MCP server URLs in configuration
- Check if rate limits are hit

#### 5. Redis Connection Failed

```
redis.exceptions.ConnectionError
```

**Solution:**
```bash
# Start Redis with Docker
docker-compose up -d redis

# Or disable Redis caching
export REDIS_ENABLED=false
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
2. **Customize Agents** - Extend agents for your use case
3. **Add MCP Servers** - Integrate additional data sources
4. **Deploy to Production** - Use the CDK stack

For more detailed documentation, see:
- [Architecture Guide](./architecture.md)
- [API Reference](./api-reference.md)
- [Deployment Guide](./deployment.md)
