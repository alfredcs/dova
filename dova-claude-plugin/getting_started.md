# Dova Claude Code Plugin — Getting Started

Use Dova's multi-agent research platform directly inside Claude Code. Search ArXiv papers, GitHub repos, HuggingFace models, run structured debates, and validate code — all without leaving your terminal.

## Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed
- Python 3.11+
- Dova installed (`pip install -e .` from the repo root)
- At least one LLM provider configured (AWS Bedrock, Anthropic API, or OpenAI)

## Quick Setup

### 1. Install Dova

```bash
git clone <repo-url> && cd dova
pip install -e .
```

### 2. Configure credentials

Copy the example and fill in your provider keys:

```bash
cp .env.example .env
```

**Minimum required** — pick one LLM provider:

```bash
# Option A: AWS Bedrock (default)
AWS_REGION=us-east-1
LLM_PRIMARY_PROVIDER=bedrock

# Option B: Anthropic API
LLM_PRIMARY_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Option C: OpenAI
LLM_PRIMARY_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

**Optional** — web search providers (DuckDuckGo works without keys):

```bash
BRAVE_API_KEY=...       # https://brave.com/search/api/
PERPLEXITY_API_KEY=...  # https://perplexity.ai/settings/api
TAVILY_API_KEY=...      # https://tavily.com
```

### 3. Verify the MCP server starts

```bash
dova mcp serve --help
```

You should see the serve command with `--transport` and `--port` options.

### 4. Load the plugin in Claude Code

```bash
claude --plugin-dir ./dova-claude-plugin
```

Claude Code will automatically start the Dova MCP server as a subprocess and make all 5 tools available.

## What You Get

### MCP Tools (available to Claude automatically)

| Tool | What it does |
|------|-------------|
| `dova_research` | Search across all sources at once (ArXiv, GitHub, HuggingFace, web) |
| `dova_search` | Search a single source — `arxiv`, `github`, `huggingface`, or `web` |
| `dova_debate` | Run a Bull vs Bear structured debate on any topic |
| `dova_validate` | Analyze code for quality, security, and correctness |
| `dova_web_search` | Search the web via Brave, Perplexity, Tavily, or DuckDuckGo |

### Skills (slash commands)

| Command | What it does |
|---------|-------------|
| `/dova-research <query>` | Research a topic and get structured results by source |
| `/dova-debate <topic>` | Get a balanced pro/con analysis with synthesis |

### Custom Agent

The **dova-researcher** agent is a research specialist that automatically uses the right combination of tools for your query. Claude Code can delegate to it for complex research tasks.

## Usage Examples

### Research a topic

```
> /dova-research transformer architectures for NLP
```

Or just ask naturally — Claude will use the tools directly:

```
> What are the latest papers on RLHF?
> Find open-source code generation models on HuggingFace
> Search GitHub for RAG framework implementations
```

### Run a debate

```
> /dova-debate Is RAG better than fine-tuning for domain adaptation?
```

### Validate code

Ask Claude to validate code and it will use `dova_validate`:

```
> Validate this function for security issues: <paste code>
```

### Targeted search

```
> Search ArXiv for "mixture of experts" papers from 2025
> Find the top HuggingFace models for text classification
```

## Configuration

### Plugin files

```
dova-claude-plugin/
  .claude-plugin/plugin.json   # Plugin manifest
  .mcp.json                    # MCP server config (stdio)
  settings.json                # Auto-allowed tool permissions
  skills/
    dova-research/SKILL.md     # /dova-research skill
    dova-debate/SKILL.md       # /dova-debate skill
  agents/
    dova-researcher.md         # Custom research agent
```

### MCP server standalone

You can also run the MCP server independently for development:

```bash
# stdio mode (default, for Claude Code)
dova mcp serve

# HTTP mode (for testing with other clients)
dova mcp serve --transport http --port 8080
```

### Manual MCP configuration

If you prefer to configure the MCP server manually instead of using the plugin, add this to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "dova": {
      "command": "dova",
      "args": ["mcp", "serve"]
    }
  }
}
```

## Troubleshooting

**"LLM router not available"**
Your LLM provider isn't configured. Check that `.env` has valid credentials for at least one provider (Bedrock, Anthropic, or OpenAI).

**"Web search service not available"**
Web search failed to initialize. DuckDuckGo should work without any API keys — check your network connection.

**MCP server won't start**
Run `dova mcp serve` directly in your terminal to see error output. Common issues:
- Missing `pip install -e .` (dova not on PATH)
- Python version < 3.11
- Missing dependencies (`pip install -e .` again)

**Tools not showing in Claude Code**
Make sure you're launching with `--plugin-dir` pointing to the correct path:
```bash
claude --plugin-dir /absolute/path/to/dova-claude-plugin
```
