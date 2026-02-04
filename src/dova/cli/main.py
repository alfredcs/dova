"""
DOVA Command Line Interface.

This module provides the CLI for interacting with the DOVA platform.
"""

import asyncio
import json
import sys
from typing import Optional

import click

from dova import __version__
from dova.config.settings import Settings


@click.group()
@click.version_option(version=__version__, prog_name="dova")
@click.option("--debug/--no-debug", default=False, help="Enable debug mode")
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    """DOVA - Deep Orchestrated Versatile Agent Platform.

    A multi-agent research automation system for ML/AI practitioners.
    """
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    ctx.obj["settings"] = Settings()

    if debug:
        import logging

        logging.basicConfig(level=logging.DEBUG)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload/--no-reload", default=True, help="Enable auto-reload")
@click.option(
    "--mode",
    type=click.Choice(["fastapi", "agentcore"]),
    default="fastapi",
    help="Runtime mode: fastapi (local dev) or agentcore (AWS deployment)",
)
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, reload: bool, mode: str) -> None:
    """Start the DOVA API server.

    Dual mode operation:
    - fastapi: Standard FastAPI server (local development, full flexibility)
    - agentcore: AWS Bedrock AgentCore Runtime (production AWS deployment)
    """
    import os

    if mode == "agentcore":
        click.echo(f"Starting DOVA in AgentCore Runtime mode on {host}:{port}")
        click.echo("Note: Requires AWS credentials and STACK_NAME environment variable")

        # Set runtime mode env var for the app
        os.environ["RUNTIME_MODE"] = "agentcore"

        try:
            from dova.runtime.agentcore_app import app

            app.run(port=port, host=host)
        except ImportError as e:
            click.echo(
                click.style(
                    f"Error: {e}\n"
                    "Install bedrock-agentcore with: pip install 'bedrock-agentcore[strands-agents]>=1.0.6'",
                    fg="red",
                )
            )
            sys.exit(1)
    else:
        import uvicorn

        click.echo(f"Starting DOVA FastAPI server on {host}:{port}")

        uvicorn.run(
            "dova.api.main:create_app",
            host=host,
            port=port,
            reload=reload,
            factory=True,
        )


@cli.command()
@click.option("--no-thinking", is_flag=True, help="Hide chain-of-thought reasoning")
@click.option("--verbose", "-v", is_flag=True, help="Show verbose output")
@click.option(
    "--orchestrator",
    "-o",
    type=click.Choice(["standard", "thinking"]),
    default="standard",
    help="Orchestrator type: standard (task-graph) or thinking (deliberation-first)",
)
@click.pass_context
def interact(ctx: click.Context, no_thinking: bool, verbose: bool, orchestrator: str) -> None:
    """Start an interactive DOVA session with continuous conversation.

    Provides a Claude Code-like experience with:
    - Chain-of-thought reasoning for sophisticated responses
    - Memory integration (recalls past interactions)
    - Multi-turn conversation with context preservation
    - Automatic tool selection (research, debate, etc.)

    Example:
        dova interact
        dova interact --no-thinking
        dova interact --verbose
        dova interact --orchestrator thinking

    In interactive mode, you can:
    - Ask research questions
    - Request analysis and debates
    - Build on previous responses
    - Use commands like /status, /help, /clear, /orchestrator
    """
    from dova.cli.interact import run_interactive_loop

    asyncio.run(run_interactive_loop(
        show_thinking=not no_thinking,
        verbose=verbose,
        orchestrator_type=orchestrator,
    ))


@cli.command()
@click.argument("query")
@click.option(
    "--sources",
    "-s",
    multiple=True,
    default=["github", "huggingface", "web"],
    help="Sources to search (arxiv, github, huggingface, web)",
)
@click.option("--max-results", "-n", default=10, help="Maximum results per source")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--format", "-f", type=click.Choice(["json", "text"]), default="text")
@click.option(
    "--reasoning",
    "-r",
    type=click.Choice(["quick", "standard", "deep", "collaborative", "tool_augmented"]),
    default="standard",
    help="Reasoning mode: quick (no reflection), standard (ReAct+reflection), deep (ensemble), collaborative (full), tool_augmented (auto tool discovery)",
)
@click.option(
    "--orchestrator",
    type=click.Choice(["standard", "thinking"]),
    default="standard",
    help="Orchestrator type: standard (task-graph) or thinking (deliberation-first)",
)
@click.pass_context
def research(
    ctx: click.Context,
    query: str,
    sources: tuple[str, ...],
    max_results: int,
    output: Optional[str],
    format: str,
    reasoning: str,
    orchestrator: str,
) -> None:
    """Run a research query through DOVA.

    Example:
        dova research "transformer architecture for NLP"
        dova research "reinforcement learning" -s arxiv -s github -n 5
        dova research "what is BERT?" --reasoning quick
        dova research "explain attention" --orchestrator thinking
    """
    from dova.agents.base import AgentTask
    from dova.agents.debate import DebateAgent
    from dova.agents.orchestrator import DOVAOrchestrator
    from dova.agents.research import ResearchAgent
    from dova.agents.synthesis import SynthesisAgent
    from dova.config.providers import create_llm_router_from_settings
    from dova.config.settings import get_settings
    from dova.tools.mcp_registry import MCPClient

    async def run_research():
        click.echo(f"Researching: {query}")

        # Initialize components
        settings = get_settings()
        llm_router = create_llm_router_from_settings()
        mcp_client = MCPClient()

        # Filter to only configured sources
        source_to_server = {
            "arxiv": "arxiv",
            "github": "github",
            "huggingface": "huggingface",
            "web": None,  # Web search uses Tavily directly, not MCP
        }
        available_sources = []
        for source in sources:
            if source == "web":
                # Web search is available if Tavily API key is configured
                if settings.mcp.tavily_api_key:
                    available_sources.append(source)
            else:
                server_name = source_to_server.get(source, source)
                server = mcp_client.registry.get_server(server_name)
                if server:
                    available_sources.append(source)

        if not available_sources:
            click.echo(click.style("No MCP servers configured. Run 'dova mcp list' to check.", fg="yellow"))
            return

        click.echo(f"Sources: {', '.join(available_sources)}")
        click.echo(f"Reasoning: {reasoning}")
        click.echo(f"Orchestrator: {orchestrator}")
        click.echo("")

        # Create specialized agents with MCP client
        research_agent = ResearchAgent(
            llm_router=llm_router,
            mcp_client=mcp_client,
            tavily_api_key=settings.mcp.tavily_api_key,
        )
        synthesis_agent = SynthesisAgent(llm_router=llm_router)
        debate_agent = DebateAgent(llm_router=llm_router, mcp_client=mcp_client)

        agents_dict = {
            "research": research_agent,
            "synthesis": synthesis_agent,
            "debate": debate_agent,
        }

        # Select orchestrator based on option
        if orchestrator == "thinking":
            from dova.agents.thinking_orchestrator import ThinkingOrchestrator

            orch = ThinkingOrchestrator(
                llm_router=llm_router,
                mcp_client=mcp_client,
                agents=agents_dict,
            )
            # ThinkingOrchestrator uses "query" task type
            task = AgentTask(
                type="query",
                params={
                    "query": query,
                    "sources": available_sources,
                },
                user_id="cli-user",
            )
        else:
            orch = DOVAOrchestrator(
                llm_router=llm_router,
                mcp_client=mcp_client,
                agents=agents_dict,
            )
            task = AgentTask(
                type="research",
                params={
                    "query": query,
                    "sources": available_sources,
                    "max_results": max_results,
                    "reasoning_mode": reasoning,
                },
                user_id="cli-user",
            )

        result = await orch.execute(task)

        if result.success:
            if format == "json":
                output_data = json.dumps(result.data, indent=2, default=str)
            else:
                output_data = format_research_results(result.data)

            if output:
                with open(output, "w") as f:
                    f.write(output_data)
                click.echo(f"Results saved to {output}")
            else:
                click.echo(output_data)
        else:
            click.echo(f"Research failed: {result.error}", err=True)
            sys.exit(1)

    asyncio.run(run_research())


def format_research_results(data: dict | None) -> str:
    """Format research results for text output."""
    lines = []

    # Handle None or empty data
    if not data:
        return "No results found."

    # Check if any actual results were found
    has_results = any([
        data.get("papers"),
        data.get("repositories"),
        data.get("models"),
        data.get("datasets"),
    ])

    if "summary" in data and data["summary"]:
        lines.append("=" * 60)
        lines.append("SUMMARY")
        lines.append("=" * 60)
        summary = data["summary"]
        if isinstance(summary, dict):
            # Handle dict summary (e.g., from synthesis agent)
            summary = summary.get("text", summary.get("content", str(summary)))
        lines.append(str(summary))
        lines.append("")

    if "papers" in data and data["papers"]:
        lines.append("=" * 60)
        lines.append(f"PAPERS ({len(data['papers'])} found)")
        lines.append("=" * 60)
        for i, paper in enumerate(data["papers"][:10], 1):  # Limit to 10
            lines.append(f"{i}. {paper.get('title', 'Unknown')}")
            if "authors" in paper:
                authors = paper["authors"][:3] if isinstance(paper["authors"], list) else [paper["authors"]]
                lines.append(f"   Authors: {', '.join(str(a) for a in authors)}")
            if "url" in paper:
                lines.append(f"   URL: {paper['url']}")
            lines.append("")

    if "repositories" in data and data["repositories"]:
        lines.append("=" * 60)
        lines.append(f"REPOSITORIES ({len(data['repositories'])} found)")
        lines.append("=" * 60)
        for i, repo in enumerate(data["repositories"][:10], 1):  # Limit to 10
            name = repo.get('name', repo.get('title', 'Unknown'))
            lines.append(f"{i}. {name}")
            desc = repo.get("description")
            if desc:
                desc_str = str(desc)[:100] if desc else ""
                if desc_str:
                    lines.append(f"   {desc_str}...")
            if "stars" in repo:
                lines.append(f"   ⭐ {repo['stars']} stars")
            if "url" in repo:
                lines.append(f"   URL: {repo['url']}")
            lines.append("")

    if "models" in data and data["models"]:
        lines.append("=" * 60)
        lines.append(f"MODELS ({len(data['models'])} found)")
        lines.append("=" * 60)
        for i, model in enumerate(data["models"][:10], 1):  # Limit to 10
            lines.append(f"{i}. {model.get('id', model.get('title', 'Unknown'))}")
            if "downloads" in model:
                lines.append(f"   Downloads: {model['downloads']:,}")
            if "pipeline_tag" in model:
                lines.append(f"   Task: {model['pipeline_tag']}")
            if "url" in model:
                lines.append(f"   URL: {model['url']}")
            lines.append("")

    if "datasets" in data and data["datasets"]:
        lines.append("=" * 60)
        lines.append(f"DATASETS ({len(data['datasets'])} found)")
        lines.append("=" * 60)
        for i, ds in enumerate(data["datasets"][:10], 1):  # Limit to 10
            lines.append(f"{i}. {ds.get('id', ds.get('title', 'Unknown'))}")
            if "downloads" in ds:
                lines.append(f"   Downloads: {ds['downloads']:,}")
            if "url" in ds:
                lines.append(f"   URL: {ds['url']}")
            lines.append("")

    if "insights" in data and data["insights"]:
        lines.append("=" * 60)
        lines.append("KEY FINDINGS")
        lines.append("=" * 60)
        for insight in data["insights"]:
            if isinstance(insight, dict):
                lines.append(f"• {insight.get('title', insight.get('summary', str(insight)))}")
                if insight.get("summary") and insight.get("title"):
                    lines.append(f"  {insight['summary']}")
            else:
                lines.append(f"• {insight}")
        lines.append("")

    if "recommendations" in data and data["recommendations"]:
        lines.append("=" * 60)
        lines.append("RECOMMENDATIONS")
        lines.append("=" * 60)
        for rec in data["recommendations"]:
            if isinstance(rec, dict):
                priority = rec.get("priority", "")
                action = rec.get("action", rec.get("recommendation", str(rec)))
                lines.append(f"→ [{priority}] {action}" if priority else f"→ {action}")
                if rec.get("details") or rec.get("description"):
                    lines.append(f"  {rec.get('details', rec.get('description', ''))}")
            else:
                lines.append(f"→ {rec}")
        lines.append("")

    if "debate" in data and data["debate"]:
        debate = data["debate"]
        lines.append("=" * 60)
        lines.append("DEBATE ANALYSIS (Bull vs Bear)")
        lines.append("=" * 60)
        if debate.get("summary"):
            lines.append(f"\n{debate['summary']}\n")
        if debate.get("bull_strengths"):
            lines.append("Strengths (Bull):")
            for s in debate["bull_strengths"][:3]:
                lines.append(f"  + {s}")
        if debate.get("bear_concerns"):
            lines.append("\nConcerns (Bear):")
            for c in debate["bear_concerns"][:3]:
                lines.append(f"  - {c}")
        if debate.get("recommendation"):
            lines.append(f"\nRecommendation: {debate['recommendation']}")
        lines.append("")

    if not has_results and not lines:
        lines.append("No results found. Try:")
        lines.append("  - Using different search terms")
        lines.append("  - Checking MCP server connectivity: dova mcp list")
        lines.append("  - Adding more sources: dova mcp add <name> --url <url>")

    return "\n".join(lines)


@cli.command()
@click.argument("code_path", type=click.Path(exists=True))
@click.option("--language", "-l", default="python", help="Programming language")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.pass_context
def validate(
    ctx: click.Context,
    code_path: str,
    language: str,
    output: Optional[str],
) -> None:
    """Validate code quality and security.

    Example:
        dova validate ./src/my_module.py
        dova validate ./project -l python -o report.json
    """
    from dova.agents.base import AgentTask
    from dova.agents.validation import ValidationAgent
    from dova.config.providers import create_llm_router_from_settings

    async def run_validation():
        click.echo(f"Validating: {code_path}")
        click.echo(f"Language: {language}")
        click.echo("")

        # Read code
        with open(code_path, "r") as f:
            code = f.read()

        llm_router = create_llm_router_from_settings()

        validator = ValidationAgent(
            llm_router=llm_router,
            mcp_client=None,
        )

        task = AgentTask(
            type="validation",
            params={
                "code": code,
                "language": language,
                "file_path": code_path,
            },
            user_id="cli-user",
        )

        result = await validator.execute(task)

        if result.success:
            output_data = json.dumps(result.data, indent=2, default=str)

            if output:
                with open(output, "w") as f:
                    f.write(output_data)
                click.echo(f"Validation report saved to {output}")
            else:
                click.echo("Validation Results:")
                click.echo("-" * 40)

                data = result.data
                if data.get("is_valid"):
                    click.echo(click.style("✓ Code is valid", fg="green"))
                else:
                    click.echo(click.style("✗ Issues found", fg="red"))

                if "quality_score" in data:
                    click.echo(f"Quality Score: {data['quality_score']:.2f}")

                if data.get("issues"):
                    click.echo("\nIssues:")
                    for issue in data["issues"]:
                        click.echo(f"  • {issue}")

                if data.get("suggestions"):
                    click.echo("\nSuggestions:")
                    for suggestion in data["suggestions"]:
                        click.echo(f"  → {suggestion}")
        else:
            click.echo(f"Validation failed: {result.error}", err=True)
            sys.exit(1)

    asyncio.run(run_validation())


@cli.command()
@click.option("--check", "-c", is_flag=True, help="Only check health status")
@click.pass_context
def health(ctx: click.Context, check: bool) -> None:
    """Check DOVA system health.

    Example:
        dova health
        dova health --check
    """
    import httpx

    settings = ctx.obj["settings"]
    api_url = f"http://localhost:{settings.api.port}"

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{api_url}/health")

            if response.status_code == 200:
                data = response.json()
                click.echo(click.style("DOVA System Health", bold=True))
                click.echo("-" * 40)
                click.echo(f"Status: {click.style(data.get('status', 'unknown'), fg='green')}")
                click.echo(f"Version: {data.get('version', 'unknown')}")

                if "components" in data:
                    click.echo("\nComponents:")
                    for name, status in data["components"].items():
                        color = "green" if status == "healthy" else "red"
                        click.echo(f"  • {name}: {click.style(status, fg=color)}")

                if check:
                    sys.exit(0 if data.get("status") == "healthy" else 1)
            else:
                click.echo(click.style(f"Health check failed: {response.status_code}", fg="red"))
                if check:
                    sys.exit(1)

    except httpx.ConnectError:
        click.echo(click.style("Could not connect to DOVA server", fg="red"))
        click.echo(f"Is the server running on {api_url}?")
        if check:
            sys.exit(1)
    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        if check:
            sys.exit(1)


@cli.command()
@click.pass_context
def config(ctx: click.Context) -> None:
    """Show current configuration."""
    settings = ctx.obj["settings"]

    click.echo(click.style("DOVA Configuration", bold=True))
    click.echo("=" * 50)

    # Environment
    click.echo(f"\nEnvironment: {settings.environment}")
    click.echo(f"Log Level: {settings.log_level}")

    # AWS
    click.echo("\nAWS:")
    click.echo(f"  Region: {settings.aws.region}")
    click.echo(f"  Bedrock Model: {settings.aws.bedrock_model_id}")

    # LLM
    click.echo("\nLLM:")
    click.echo(f"  Default Provider: {settings.llm.default_provider}")
    click.echo(f"  Temperature: {settings.llm.temperature}")
    click.echo(f"  Max Tokens: {settings.llm.max_tokens}")

    # MCP
    click.echo("\nMCP Servers:")
    if settings.mcp.arxiv_enabled:
        click.echo("  • arxiv")
    if settings.mcp.github_enabled:
        click.echo("  • github")
    if settings.mcp.huggingface_enabled:
        click.echo("  • huggingface")

    # API
    click.echo("\nAPI:")
    click.echo(f"  Host: {settings.api.host}")
    click.echo(f"  Port: {settings.api.port}")

    # Advanced Intelligence
    click.echo("\nAdvanced Intelligence:")
    click.echo(f"  Thinking Level: {settings.thinking.default_level}")
    click.echo(f"  Auto-Select Thinking: {settings.thinking.auto_select_enabled}")
    click.echo(f"  Auto-Evaluate: {settings.evaluation.auto_evaluate_responses}")
    click.echo(f"  Min Confidence: {settings.evaluation.min_confidence_threshold}")

    # Session
    click.echo("\nSession:")
    click.echo(f"  Stale After: {settings.session.stale_after_seconds}s")
    click.echo(f"  Expire After: {settings.session.expire_after_seconds}s")

    # Heartbeat
    click.echo("\nHeartbeat:")
    click.echo(f"  Enabled: {settings.heartbeat.enabled}")


@cli.group()
def profile() -> None:
    """Manage user profiles."""
    pass


@profile.command("show")
@click.argument("user_id")
@click.pass_context
def profile_show(ctx: click.Context, user_id: str) -> None:
    """Show user profile."""
    click.echo(f"Profile for user: {user_id}")
    click.echo("(Profile retrieval not implemented in CLI)")


@profile.command("update")
@click.argument("user_id")
@click.option("--interests", "-i", multiple=True, help="User interests")
@click.option(
    "--expertise",
    "-e",
    type=click.Choice(["beginner", "intermediate", "advanced", "expert"]),
    help="Expertise level",
)
@click.pass_context
def profile_update(
    ctx: click.Context,
    user_id: str,
    interests: tuple[str, ...],
    expertise: Optional[str],
) -> None:
    """Update user profile."""
    click.echo(f"Updating profile for user: {user_id}")
    if interests:
        click.echo(f"  Interests: {', '.join(interests)}")
    if expertise:
        click.echo(f"  Expertise: {expertise}")
    click.echo("(Profile update not implemented in CLI)")


@cli.command()
@click.pass_context
def models(ctx: click.Context) -> None:
    """Show LLM model configuration and tier mapping.

    Displays which models are used for different task types,
    organized by capability tier (basic, standard, advanced, reasoning).
    """
    from dova.config.providers import (
        DEFAULT_ANTHROPIC_MODELS,
        DEFAULT_BEDROCK_MODELS,
        DEFAULT_OPENAI_MODELS,
        TASK_TIER_MAPPING,
        ModelTier,
        create_llm_router_from_settings,
    )

    click.echo(click.style("\n=== DOVA Model Configuration ===\n", bold=True))

    # Task to Tier mapping
    click.echo(click.style("Task → Tier Mapping:", bold=True))
    click.echo("-" * 50)
    tier_colors = {
        ModelTier.BASIC: "green",
        ModelTier.STANDARD: "yellow",
        ModelTier.ADVANCED: "red",
        ModelTier.REASONING: "magenta",
    }
    for task_type, tier in TASK_TIER_MAPPING.items():
        color = tier_colors.get(tier, "white")
        click.echo(f"  {task_type.value:20s} → {click.style(tier.value, fg=color)}")

    # Default models by provider
    click.echo(click.style("\n\nDefault Models by Provider:", bold=True))
    click.echo("-" * 60)

    providers = [
        ("Bedrock (AWS)", DEFAULT_BEDROCK_MODELS, "cyan"),
        ("Anthropic", DEFAULT_ANTHROPIC_MODELS, "blue"),
        ("OpenAI", DEFAULT_OPENAI_MODELS, "green"),
    ]

    for provider_name, models, color in providers:
        click.echo(click.style(f"\n{provider_name}:", fg=color, bold=True))
        for tier, model_id in models.items():
            tier_color = tier_colors.get(tier, "white")
            tier_str = f"{tier.value:12s}"
            click.echo(f"  {click.style(tier_str, fg=tier_color)} → {model_id}")

    # Currently configured router
    click.echo(click.style("\n\nCurrently Active Configuration:", bold=True))
    click.echo("-" * 60)

    try:
        router = create_llm_router_from_settings()
        for name, provider in router.providers.items():
            click.echo(click.style(f"\n{name.upper()} (enabled):", fg="green", bold=True))
            for task_type, model_config in provider.config.models.items():
                tier = TASK_TIER_MAPPING.get(task_type, ModelTier.STANDARD)
                tier_color = tier_colors.get(tier, "white")
                tier_label = click.style(f"{tier.value:8s}", fg=tier_color)
                click.echo(f"  {task_type.value:20s} [{tier_label}] → {model_config.model_id}")
    except Exception as e:
        click.echo(click.style(f"  Error loading router: {e}", fg="red"))

    # Environment variable hints
    click.echo(click.style("\n\nCustomize with Environment Variables:", bold=True))
    click.echo("-" * 60)
    click.echo("""
  # Bedrock tier models
  export BEDROCK_MODEL_BASIC=us.anthropic.claude-haiku-4-5-20251001-v1:0
  export BEDROCK_MODEL_STANDARD=us.anthropic.claude-sonnet-4-20250514-v1:0
  export BEDROCK_MODEL_ADVANCED=us.anthropic.claude-sonnet-4-20250514-v1:0

  # Anthropic tier models
  export ANTHROPIC_MODEL_BASIC=claude-haiku-3-5-20241022
  export ANTHROPIC_MODEL_ADVANCED=claude-sonnet-4-20250514

  # OpenAI tier models
  export OPENAI_MODEL_BASIC=gpt-4o-mini
  export OPENAI_MODEL_ADVANCED=gpt-4o
    """)


@cli.group()
def mcp() -> None:
    """Manage MCP server configurations."""
    pass


@mcp.command("add")
@click.argument("name")
@click.option("--url", "-u", required=True, help="MCP server URL")
@click.option(
    "--type",
    "-t",
    "server_type",
    default="http",
    type=click.Choice(["http", "sse"]),
    help="Transport type",
)
@click.option(
    "--header",
    "-H",
    multiple=True,
    help="Auth token or header. Use just the token value, or 'Key: Value' format",
)
def mcp_add(name: str, url: str, server_type: str, header: tuple[str, ...]) -> None:
    """Add or update an MCP server configuration.

    Example:
        dova mcp add arxiv --url http://localhost:8084/mcp
        dova mcp add github --url https://api.github.com/mcp -H ghp_xxxtoken
        dova mcp add custom --url https://example.com/mcp -H "X-Api-Key: mykey"
    """
    from dova.config.mcp_servers import add_mcp_server

    # Parse headers - support both token shorthand and Key: Value format
    headers = {}
    for h in header:
        if ":" in h:
            # Full header format: "Key: Value"
            key, value = h.split(":", 1)
            headers[key.strip()] = value.strip()
        else:
            # Token shorthand: just the token value -> "Authorization: Bearer token"
            headers["Authorization"] = f"Bearer {h}"

    add_mcp_server(name, url, server_type, headers if headers else None)
    click.echo(f"Added MCP server: {name}")
    click.echo(f"  URL: {url}")
    click.echo(f"  Type: {server_type}")
    if headers:
        click.echo(f"  Headers: {len(headers)} configured")


@mcp.command("remove")
@click.argument("name")
def mcp_remove(name: str) -> None:
    """Remove an MCP server configuration.

    Example:
        dova mcp remove arxiv
    """
    from dova.config.mcp_servers import remove_mcp_server

    if remove_mcp_server(name):
        click.echo(f"Removed MCP server: {name}")
    else:
        click.echo(f"MCP server not found: {name}", err=True)


@mcp.command("list")
@click.option("--no-check", is_flag=True, help="Skip health checks")
def mcp_list(no_check: bool) -> None:
    """List all configured MCP servers with health status.

    Example:
        dova mcp list
        dova mcp list --no-check
    """
    import httpx

    from dova.config.mcp_servers import list_mcp_servers, get_dova_config_path, load_managed_mcp_servers

    servers = list_mcp_servers()

    # Also include managed repos (like arxiv-mcp-server)
    managed = load_managed_mcp_servers()
    managed_names = set()
    for name, config in managed.items():
        if name not in servers:
            servers[name] = {
                "type": "stdio",
                "command": config.command,
                "managed": True,
            }
            managed_names.add(name)

    if not servers:
        click.echo(f"No MCP servers configured in {get_dova_config_path()}")
        click.echo("\nAdd servers with: dova mcp add <name> --url <url>")
        click.echo("Or setup managed repos with: dova mcp setup")
        return

    async def check_server_health(name: str, config: dict) -> tuple[str, bool, str]:
        """Check if an MCP server is reachable."""
        # STDIO servers don't have URLs to check
        if config.get("type") == "stdio":
            return (name, True, "STDIO (local)")

        url = config.get("url")
        if not url:
            return (name, False, "No URL configured")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",  # Required by some MCP servers (e.g., HuggingFace)
        }
        headers.update(config.get("headers", {}))

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # Send a simple JSON-RPC request to check connectivity
                response = await client.post(
                    url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "dova", "version": "0.1.0"},
                        },
                    },
                    headers=headers,
                )
                if response.status_code == 200:
                    return (name, True, "OK")
                else:
                    return (name, False, f"HTTP {response.status_code}")
        except httpx.ConnectError:
            return (name, False, "Connection refused")
        except httpx.TimeoutException:
            return (name, False, "Timeout")
        except Exception as e:
            return (name, False, str(e)[:30])

    async def check_all_servers():
        import asyncio

        tasks = [check_server_health(name, config) for name, config in servers.items()]
        return await asyncio.gather(*tasks)

    click.echo(click.style("Configured MCP Servers", bold=True))
    click.echo(f"Config file: {get_dova_config_path()}")
    click.echo("-" * 50)

    # Run health checks if not disabled
    health_results = {}
    if not no_check:
        click.echo("Checking connectivity...")
        results = asyncio.run(check_all_servers())
        health_results = {name: (ok, msg) for name, ok, msg in results}

    for name, config in servers.items():
        # Determine status indicator
        if no_check:
            status = click.style("?", fg="yellow")
            status_msg = ""
        elif name in health_results:
            ok, msg = health_results[name]
            if ok:
                status = click.style("✓", fg="green")
                status_msg = ""
            else:
                status = click.style("✗", fg="red")
                status_msg = f" ({msg})"
        else:
            status = click.style("?", fg="yellow")
            status_msg = ""

        click.echo(f"\n{status} {click.style(name, bold=True)}{status_msg}")
        server_type = config.get("type", "http")
        click.echo(f"  Type: {server_type}")
        if server_type == "stdio":
            cmd = config.get("command", "N/A")
            # Truncate long commands
            if len(cmd) > 60:
                cmd = cmd[:57] + "..."
            click.echo(f"  Command: {cmd}")
            if config.get("managed"):
                click.echo(f"  Source: Managed repo (dova mcp setup)")
        else:
            click.echo(f"  URL: {config.get('url', 'N/A')}")
        if config.get("headers"):
            click.echo(f"  Headers: {len(config['headers'])} configured")


@mcp.command("show")
def mcp_show() -> None:
    """Show the full MCP configuration file.

    Example:
        dova mcp show
    """
    import json
    import os

    from dova.config.mcp_servers import get_dova_config_path

    config_path = get_dova_config_path()

    if not os.path.exists(config_path):
        click.echo(f"Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    click.echo(json.dumps(config, indent=2))


@mcp.command("test")
@click.argument("name")
@click.option("--tool", "-t", default=None, help="Tool name to test (optional)")
def mcp_test(name: str, tool: Optional[str]) -> None:
    """Test connectivity to an MCP server.

    Example:
        dova mcp test huggingface
        dova mcp test huggingface --tool model_search
    """
    from dova.tools.mcp_registry import MCPClient

    async def run_test():
        client = MCPClient()

        server_config = client.registry.get_server(name)
        if not server_config:
            click.echo(f"MCP server not found: {name}", err=True)
            click.echo("\nConfigured servers:")
            for s in client.registry.servers:
                click.echo(f"  - {s}")
            return

        click.echo(f"Testing MCP server: {name}")
        click.echo(f"  URL: {server_config.url}")
        click.echo(f"  Transport: {server_config.transport.value}")

        if tool:
            click.echo(f"\nInvoking tool: {tool}")
            try:
                # Use appropriate test params for different servers
                if name == "arxiv":
                    test_params = {"query": "machine learning", "max_results": 1}
                elif name == "huggingface":
                    test_params = {"query": "test", "limit": 1}
                else:
                    test_params = {"query": "test", "limit": 1}
                result = await client.invoke(name, tool, test_params)
                click.echo(click.style("✓ Success", fg="green"))
                # Truncate long results
                result_str = str(result)
                if len(result_str) > 500:
                    result_str = result_str[:500] + "..."
                click.echo(f"Result: {result_str}")
            except Exception as e:
                click.echo(click.style(f"✗ Failed: {e}", fg="red"))
        else:
            click.echo(click.style("✓ Server configured", fg="green"))

    asyncio.run(run_test())


@mcp.command("setup")
@click.option("--name", "-n", default=None, help="Specific repo name to setup (default: all)")
def mcp_setup(name: Optional[str]) -> None:
    """Clone and setup MCP server repositories.

    This clones MCP server repos (like arxiv-mcp-server) locally
    and installs their dependencies.

    Example:
        dova mcp setup              # Setup all repos
        dova mcp setup --name arxiv # Setup specific repo
    """
    from dova.services.mcp_repo_manager import get_mcp_repo_manager

    async def run_setup():
        manager = get_mcp_repo_manager()

        if name:
            config = manager.get_repo(name)
            if not config:
                click.echo(f"Unknown MCP repo: {name}", err=True)
                click.echo("\nAvailable repos:")
                for repo in manager.list_repos():
                    click.echo(f"  - {repo.name}: {repo.repo_url}")
                return

            click.echo(f"Setting up MCP repo: {name}")
            click.echo(f"  URL: {config.repo_url}")
            click.echo(f"  Path: {config.local_path}")

            result = await manager.setup_repo(name)
            if result:
                click.echo(click.style("✓ Setup successful", fg="green"))
            else:
                click.echo(click.style("✗ Setup failed", fg="red"))
        else:
            click.echo("Setting up all MCP repos...")
            results = await manager.setup_all()
            for repo_name, success in results.items():
                status = click.style("✓", fg="green") if success else click.style("✗", fg="red")
                click.echo(f"  {status} {repo_name}")

    asyncio.run(run_setup())


@mcp.command("update")
@click.option("--name", "-n", default=None, help="Specific repo name to update (default: all)")
def mcp_update(name: Optional[str]) -> None:
    """Git pull updates for MCP server repositories.

    This updates locally cloned MCP server repos to the latest version.

    Example:
        dova mcp update              # Update all repos
        dova mcp update --name arxiv # Update specific repo
    """
    from dova.services.mcp_repo_manager import get_mcp_repo_manager

    async def run_update():
        manager = get_mcp_repo_manager()

        if name:
            config = manager.get_repo(name)
            if not config:
                click.echo(f"Unknown MCP repo: {name}", err=True)
                return

            if not manager.is_repo_installed(name):
                click.echo(f"Repo not installed: {name}", err=True)
                click.echo("Run 'dova mcp setup' first.")
                return

            click.echo(f"Updating MCP repo: {name}")
            result = await manager.setup_repo(name)
            if result:
                click.echo(click.style("✓ Update successful", fg="green"))
            else:
                click.echo(click.style("✗ Update failed", fg="red"))
        else:
            click.echo("Updating all MCP repos...")
            results = await manager.update_all()
            for repo_name, success in results.items():
                status = click.style("✓", fg="green") if success else click.style("✗", fg="red")
                click.echo(f"  {status} {repo_name}")

    asyncio.run(run_update())


@mcp.command("repos")
def mcp_repos() -> None:
    """List managed MCP server repositories.

    Example:
        dova mcp repos
    """
    from dova.services.mcp_repo_manager import get_mcp_repo_manager

    manager = get_mcp_repo_manager()
    repos = manager.list_repos()

    if not repos:
        click.echo("No MCP repos configured.")
        return

    click.echo("Managed MCP server repositories:\n")
    for repo in repos:
        installed = manager.is_repo_installed(repo.name)
        status = click.style("✓ Installed", fg="green") if installed else click.style("✗ Not installed", fg="yellow")

        click.echo(f"  {repo.name}")
        click.echo(f"    Status: {status}")
        click.echo(f"    URL: {repo.repo_url}")
        click.echo(f"    Path: {repo.local_path}")
        if repo.last_updated:
            click.echo(f"    Last Updated: {repo.last_updated.isoformat()}")
        click.echo("")


# =============================================================================
# AWS Commands
# =============================================================================


@cli.group()
def aws() -> None:
    """AWS setup and management commands.

    Automates setup of AWS services required for DOVA AgentCore:
    - Cognito (OAuth2 authentication)
    - IAM (policies and roles)
    - SSM Parameter Store (configuration)
    - Secrets Manager (credentials)
    - Bedrock (model access)

    Example:
        dova aws setup --stack-name my-dova-stack
        dova aws validate --stack-name my-dova-stack
        dova aws teardown --stack-name my-dova-stack
    """
    pass


@aws.command("setup")
@click.option(
    "--stack-name",
    "-n",
    default=None,
    help="Stack name for AWS resources (default: auto-generated)",
)
@click.option(
    "--region",
    "-r",
    default="us-east-1",
    help="AWS region (default: us-east-1)",
)
@click.option(
    "--no-bedrock",
    is_flag=True,
    default=False,
    help="Skip Bedrock policy creation",
)
@click.option(
    "--no-agentcore",
    is_flag=True,
    default=False,
    help="Skip AgentCore policy creation",
)
@click.option(
    "--gateway-url",
    default=None,
    help="AgentCore Gateway URL (optional)",
)
@click.option(
    "--memory-id",
    default=None,
    help="AgentCore Memory ID (optional)",
)
@click.option(
    "--env-file",
    default=".env.aws",
    help="Path to generate environment file (default: .env.aws)",
)
@click.option(
    "--skip-validation",
    is_flag=True,
    default=False,
    help="Skip pre-flight validation (not recommended)",
)
def aws_setup(
    stack_name: Optional[str],
    region: str,
    no_bedrock: bool,
    no_agentcore: bool,
    gateway_url: Optional[str],
    memory_id: Optional[str],
    env_file: str,
    skip_validation: bool,
) -> None:
    """Set up AWS services for DOVA AgentCore.

    This command automates the creation of all AWS resources required
    for running DOVA with AgentCore Runtime:

    \b
    1. IAM Role and Policies
       - Bedrock model invocation
       - AgentCore Memory/Gateway access
       - SSM Parameter Store access
       - Secrets Manager access

    \b
    2. Cognito User Pool
       - OAuth2 authentication
       - Machine-to-machine client credentials
       - API scopes for gateway access

    \b
    3. Configuration Storage
       - SSM parameters for public config
       - Secrets Manager for client secrets

    \b
    Prerequisites:
    - AWS credentials configured (env vars, ~/.aws/credentials, or IAM role)
    - IAM permissions to create resources (see 'dova aws permissions')

    Example:
        dova aws setup --stack-name my-app --region us-west-2

    After setup, source the generated .env.aws file:
        source .env.aws && dova serve --mode agentcore
    """
    import secrets as sec

    from dova.aws.setup import AWSSetup, SetupConfig, format_setup_result

    # Generate stack name if not provided
    if not stack_name:
        stack_name = f"dova-{sec.token_hex(4)}"
        click.echo(f"Using auto-generated stack name: {stack_name}")

    click.echo(f"\nSetting up AWS resources for DOVA")
    click.echo(f"Stack: {stack_name}")
    click.echo(f"Region: {region}")
    click.echo("=" * 50)

    config = SetupConfig(
        stack_name=stack_name,
        region=region,
        include_bedrock=not no_bedrock,
        include_agentcore=not no_agentcore,
        gateway_url=gateway_url,
        memory_id=memory_id,
        env_file_path=env_file,
    )

    setup = AWSSetup(config)

    # Show progress
    click.echo("\nPhases:")
    click.echo("  1. Validating credentials...")
    result = setup.run(skip_validation=skip_validation)

    # Display result
    click.echo(format_setup_result(result))

    if result.success:
        click.echo(click.style("\n✓ AWS setup complete!", fg="green"))
        sys.exit(0)
    else:
        click.echo(click.style("\n✗ AWS setup failed", fg="red"))
        sys.exit(1)


@aws.command("validate")
@click.option(
    "--stack-name",
    "-n",
    required=True,
    help="Stack name to validate",
)
@click.option(
    "--region",
    "-r",
    default="us-east-1",
    help="AWS region (default: us-east-1)",
)
def aws_validate(stack_name: str, region: str) -> None:
    """Validate AWS setup for DOVA.

    Checks all AWS components are properly configured:
    - AWS credentials
    - Cognito User Pool and App Client
    - IAM Role and Policies
    - SSM Parameters
    - Secrets Manager secrets
    - Bedrock model access

    Example:
        dova aws validate --stack-name my-dova-stack
    """
    from dova.aws.validators import AWSValidator, format_validation_result

    click.echo(f"\nValidating AWS setup for stack: {stack_name}")
    click.echo(f"Region: {region}")
    click.echo("=" * 50)

    validator = AWSValidator(region)
    result = validator.validate_complete_setup(stack_name)

    click.echo(format_validation_result(result))

    if result.valid:
        click.echo(click.style("\n✓ All checks passed!", fg="green"))
        sys.exit(0)
    else:
        click.echo(click.style("\n✗ Validation failed", fg="red"))
        sys.exit(1)


@aws.command("teardown")
@click.option(
    "--stack-name",
    "-n",
    required=True,
    help="Stack name to teardown",
)
@click.option(
    "--region",
    "-r",
    default="us-east-1",
    help="AWS region (default: us-east-1)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt",
)
def aws_teardown(stack_name: str, region: str, force: bool) -> None:
    """Remove all AWS resources for a DOVA stack.

    WARNING: This will permanently delete:
    - Cognito User Pool (and all users)
    - IAM Role and Policies
    - SSM Parameters
    - Secrets Manager secrets

    This action cannot be undone!

    Example:
        dova aws teardown --stack-name my-dova-stack --force
    """
    from dova.aws.setup import AWSSetup, SetupConfig

    if not force:
        click.echo(f"\nThis will permanently delete all AWS resources for stack: {stack_name}")
        click.echo("This action cannot be undone!\n")

        if not click.confirm("Are you sure you want to continue?"):
            click.echo("Aborted.")
            sys.exit(0)

    click.echo(f"\nTearing down AWS resources for stack: {stack_name}")
    click.echo(f"Region: {region}")
    click.echo("=" * 50)

    config = SetupConfig(stack_name=stack_name, region=region)
    setup = AWSSetup(config)

    result = setup.teardown()

    if result.success:
        click.echo(click.style("\n✓ Teardown complete!", fg="green"))
        sys.exit(0)
    else:
        click.echo(click.style("\n✗ Teardown failed", fg="red"))
        for error in result.errors:
            click.echo(f"  - {error}")
        sys.exit(1)


@aws.command("permissions")
def aws_permissions() -> None:
    """Show required IAM permissions for AWS setup.

    Lists all IAM actions required to run 'dova aws setup'.
    You can use this to create a custom IAM policy for setup.

    Example:
        dova aws permissions
    """
    import json

    from dova.aws.iam import get_required_setup_permissions

    permissions = get_required_setup_permissions()

    click.echo("\nRequired IAM permissions for DOVA AWS setup:")
    click.echo("=" * 50)

    # Group by service
    services: dict = {}
    for perm in permissions:
        service = perm.split(":")[0]
        if service not in services:
            services[service] = []
        services[service].append(perm)

    for service, perms in sorted(services.items()):
        click.echo(f"\n{service}:")
        for perm in perms:
            click.echo(f"  - {perm}")

    # Output as policy
    click.echo("\n\nIAM Policy JSON (copy this to create a custom policy):")
    click.echo("-" * 50)

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "DOVASetupPermissions",
                "Effect": "Allow",
                "Action": permissions,
                "Resource": "*",
            }
        ],
    }
    click.echo(json.dumps(policy, indent=2))


@aws.command("env")
@click.option(
    "--stack-name",
    "-n",
    required=True,
    help="Stack name to generate env for",
)
@click.option(
    "--region",
    "-r",
    default="us-east-1",
    help="AWS region (default: us-east-1)",
)
@click.option(
    "--output",
    "-o",
    default=".env.aws",
    help="Output file path (default: .env.aws)",
)
def aws_env(stack_name: str, region: str, output: str) -> None:
    """Generate environment file from stored AWS configuration.

    Fetches configuration from SSM/Secrets Manager and generates
    a .env file for local development or deployment.

    Example:
        dova aws env --stack-name my-stack --output .env.local
    """
    from dova.aws.parameters import ParameterManager

    click.echo(f"\nGenerating environment file for stack: {stack_name}")

    params = ParameterManager(region)

    if params.generate_env_file(stack_name, output):
        click.echo(click.style(f"\n✓ Environment file generated: {output}", fg="green"))
        click.echo(f"\nUsage: source {output}")
        sys.exit(0)
    else:
        click.echo(click.style("\n✗ Failed to generate environment file", fg="red"))
        sys.exit(1)


@aws.command("deploy")
@click.option(
    "--stack-name",
    "-n",
    required=True,
    help="Stack name for deployment (must match existing setup)",
)
@click.option(
    "--region",
    "-r",
    default="us-east-1",
    help="AWS region (default: us-east-1)",
)
@click.option(
    "--memory",
    "-m",
    default=1024,
    type=int,
    help="Lambda memory in MB (default: 1024)",
)
@click.option(
    "--timeout",
    "-t",
    default=300,
    type=int,
    help="Lambda timeout in seconds (default: 300)",
)
@click.option(
    "--enable-cognito",
    is_flag=True,
    default=False,
    help="Enable Cognito authentication for API Gateway",
)
@click.option(
    "--no-cors",
    is_flag=True,
    default=False,
    help="Disable CORS on API Gateway",
)
def aws_deploy(
    stack_name: str,
    region: str,
    memory: int,
    timeout: int,
    enable_cognito: bool,
    no_cors: bool,
) -> None:
    """Deploy DOVA as a Lambda function with API Gateway.

    This command deploys DOVA to AWS Lambda behind API Gateway,
    allowing you to run DOVA as a serverless application.

    \b
    Prerequisites:
    - Run 'dova aws setup' first to create IAM roles and other resources
    - AWS credentials with deployment permissions

    \b
    What this creates:
    - Lambda function with DOVA code
    - API Gateway REST API with /invocations endpoint
    - S3 bucket for deployment artifacts

    Example:
        dova aws deploy --stack-name my-dova-stack --region us-west-2
        dova aws deploy --stack-name my-app --memory 2048 --timeout 600

    After deployment:
        curl -X POST https://<api-id>.execute-api.<region>.amazonaws.com/prod/invocations \\
          -H "Content-Type: application/json" \\
          -d '{"prompt": "What is BERT?"}'
    """
    from dova.aws.deploy import DeployConfig, DeployManager, format_deploy_result

    click.echo("\nDeploying DOVA to Lambda")
    click.echo(f"Stack: {stack_name}")
    click.echo(f"Region: {region}")
    click.echo(f"Memory: {memory} MB")
    click.echo(f"Timeout: {timeout}s")
    click.echo("=" * 50)

    config = DeployConfig(
        stack_name=stack_name,
        region=region,
        lambda_memory=memory,
        lambda_timeout=timeout,
        enable_cognito=enable_cognito,
        enable_cors=not no_cors,
    )

    deploy_manager = DeployManager(config)

    # Show progress
    click.echo("\nPhases:")
    click.echo("  1. Packaging Lambda code...")
    result = deploy_manager.deploy()

    # Display result
    click.echo(format_deploy_result(result))

    if result.success:
        click.echo(click.style("\n✓ Deployment complete!", fg="green"))
        sys.exit(0)
    else:
        click.echo(click.style("\n✗ Deployment failed", fg="red"))
        sys.exit(1)


@aws.command("status")
@click.option(
    "--stack-name",
    "-n",
    required=True,
    help="Stack name to check status for",
)
@click.option(
    "--region",
    "-r",
    default="us-east-1",
    help="AWS region (default: us-east-1)",
)
def aws_status(stack_name: str, region: str) -> None:
    """Check deployment status for a DOVA stack.

    Shows the current state of a DOVA Lambda deployment including:
    - CloudFormation stack status
    - Lambda function ARN
    - API Gateway endpoint URL

    Example:
        dova aws status --stack-name my-dova-stack
    """
    from dova.aws.deploy import DeployConfig, DeployManager, format_deploy_status

    click.echo(f"\nChecking deployment status for stack: {stack_name}")
    click.echo(f"Region: {region}")
    click.echo("=" * 50)

    config = DeployConfig(stack_name=stack_name, region=region)
    deploy_manager = DeployManager(config)

    status = deploy_manager.get_status()
    click.echo(format_deploy_status(status))

    if status:
        if status.status in ("CREATE_COMPLETE", "UPDATE_COMPLETE"):
            click.echo(click.style("\n✓ Deployment is active", fg="green"))
            sys.exit(0)
        elif "_IN_PROGRESS" in status.status:
            click.echo(click.style("\n⏳ Deployment in progress", fg="yellow"))
            sys.exit(0)
        else:
            click.echo(click.style(f"\n✗ Deployment status: {status.status}", fg="red"))
            sys.exit(1)
    else:
        click.echo(click.style("\n✗ No deployment found", fg="yellow"))
        click.echo("Run 'dova aws deploy' to create a deployment.")
        sys.exit(1)


def main() -> None:
    """Main entry point for CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
