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
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, reload: bool) -> None:
    """Start the DOVA API server."""
    import uvicorn

    from dova.api.main import create_app

    click.echo(f"Starting DOVA server on {host}:{port}")

    uvicorn.run(
        "dova.api.main:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


@cli.command()
@click.argument("query")
@click.option(
    "--sources",
    "-s",
    multiple=True,
    default=["arxiv", "github", "huggingface"],
    help="Sources to search",
)
@click.option("--max-results", "-n", default=10, help="Maximum results per source")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--format", "-f", type=click.Choice(["json", "text"]), default="text")
@click.pass_context
def research(
    ctx: click.Context,
    query: str,
    sources: tuple[str, ...],
    max_results: int,
    output: Optional[str],
    format: str,
) -> None:
    """Run a research query through DOVA.

    Example:
        dova research "transformer architecture for NLP"
        dova research "reinforcement learning" -s arxiv -s github -n 5
    """
    from dova.agents.base import AgentTask
    from dova.agents.orchestrator import DOVAOrchestrator
    from dova.config.providers import LLMRouter

    async def run_research():
        click.echo(f"Researching: {query}")
        click.echo(f"Sources: {', '.join(sources)}")
        click.echo("")

        settings = ctx.obj["settings"]

        # Initialize components
        # Note: In real usage, these would be properly configured
        llm_router = LLMRouter(providers={})  # Would be configured from settings

        orchestrator = DOVAOrchestrator(
            llm_router=llm_router,
            mcp_client=None,  # Would be configured
        )

        task = AgentTask(
            type="research",
            params={
                "query": query,
                "sources": list(sources),
                "max_results": max_results,
            },
            user_id="cli-user",
        )

        result = await orchestrator.execute(task)

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


def format_research_results(data: dict) -> str:
    """Format research results for text output."""
    lines = []

    if "summary" in data:
        lines.append("=" * 60)
        lines.append("SUMMARY")
        lines.append("=" * 60)
        lines.append(data["summary"])
        lines.append("")

    if "papers" in data and data["papers"]:
        lines.append("=" * 60)
        lines.append("PAPERS")
        lines.append("=" * 60)
        for i, paper in enumerate(data["papers"], 1):
            lines.append(f"{i}. {paper.get('title', 'Unknown')}")
            if "authors" in paper:
                lines.append(f"   Authors: {', '.join(paper['authors'][:3])}")
            if "id" in paper:
                lines.append(f"   ID: {paper['id']}")
            lines.append("")

    if "repositories" in data and data["repositories"]:
        lines.append("=" * 60)
        lines.append("REPOSITORIES")
        lines.append("=" * 60)
        for i, repo in enumerate(data["repositories"], 1):
            lines.append(f"{i}. {repo.get('name', 'Unknown')}")
            if "stars" in repo:
                lines.append(f"   Stars: {repo['stars']}")
            if "url" in repo:
                lines.append(f"   URL: {repo['url']}")
            lines.append("")

    if "models" in data and data["models"]:
        lines.append("=" * 60)
        lines.append("MODELS")
        lines.append("=" * 60)
        for i, model in enumerate(data["models"], 1):
            lines.append(f"{i}. {model.get('id', 'Unknown')}")
            if "downloads" in model:
                lines.append(f"   Downloads: {model['downloads']:,}")
            if "task" in model:
                lines.append(f"   Task: {model['task']}")
            lines.append("")

    if "insights" in data and data["insights"]:
        lines.append("=" * 60)
        lines.append("INSIGHTS")
        lines.append("=" * 60)
        for insight in data["insights"]:
            lines.append(f"• {insight}")
        lines.append("")

    if "recommendations" in data and data["recommendations"]:
        lines.append("=" * 60)
        lines.append("RECOMMENDATIONS")
        lines.append("=" * 60)
        for rec in data["recommendations"]:
            lines.append(f"→ {rec}")
        lines.append("")

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
    from dova.config.providers import LLMRouter

    async def run_validation():
        click.echo(f"Validating: {code_path}")
        click.echo(f"Language: {language}")
        click.echo("")

        # Read code
        with open(code_path, "r") as f:
            code = f.read()

        llm_router = LLMRouter(providers={})

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
    click.echo(f"  Profile: {settings.aws.profile or 'default'}")

    # LLM
    click.echo("\nLLM:")
    click.echo(f"  Primary Provider: {settings.llm.primary_provider}")
    click.echo(f"  Fallback Chain: {', '.join(settings.llm.fallback_chain)}")

    # MCP
    click.echo("\nMCP Servers:")
    for server in settings.mcp.enabled_servers:
        click.echo(f"  • {server}")

    # API
    click.echo("\nAPI:")
    click.echo(f"  Host: {settings.api.host}")
    click.echo(f"  Port: {settings.api.port}")


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


def main() -> None:
    """Main entry point for CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
