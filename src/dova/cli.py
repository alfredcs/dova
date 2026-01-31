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
    from dova.agents.research import ResearchAgent
    from dova.agents.synthesis import SynthesisAgent
    from dova.config.providers import create_llm_router_from_settings
    from dova.tools.mcp_registry import MCPClient

    async def run_research():
        click.echo(f"Researching: {query}")

        # Initialize components
        llm_router = create_llm_router_from_settings()
        mcp_client = MCPClient()

        # Filter to only configured sources
        source_to_server = {
            "arxiv": "arxiv",
            "github": "github",
            "huggingface": "huggingface",
        }
        available_sources = []
        for source in sources:
            server_name = source_to_server.get(source, source)
            server = mcp_client.registry.get_server(server_name)
            if server:
                available_sources.append(source)

        if not available_sources:
            click.echo(click.style("No MCP servers configured. Run 'dova mcp list' to check.", fg="yellow"))
            return

        click.echo(f"Sources: {', '.join(available_sources)}")
        click.echo("")

        # Create specialized agents with MCP client
        research_agent = ResearchAgent(llm_router=llm_router, mcp_client=mcp_client)
        synthesis_agent = SynthesisAgent(llm_router=llm_router)

        orchestrator = DOVAOrchestrator(
            llm_router=llm_router,
            mcp_client=mcp_client,
            agents={
                "research": research_agent,
                "synthesis": synthesis_agent,
            },
        )

        task = AgentTask(
            type="research",
            params={
                "query": query,
                "sources": available_sources,
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

    from dova.config.mcp_servers import list_mcp_servers, get_dova_config_path

    servers = list_mcp_servers()

    if not servers:
        click.echo(f"No MCP servers configured in {get_dova_config_path()}")
        click.echo("\nAdd servers with: dova mcp add <name> --url <url>")
        return

    async def check_server_health(name: str, config: dict) -> tuple[str, bool, str]:
        """Check if an MCP server is reachable."""
        url = config.get("url")
        if not url:
            return (name, False, "No URL configured")

        headers = {"Content-Type": "application/json"}
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
        click.echo(f"  Type: {config.get('type', 'http')}")
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
                result = await client.invoke(name, tool, {"query": "test", "limit": 1})
                click.echo(click.style("✓ Success", fg="green"))
                click.echo(f"Result: {result}")
            except Exception as e:
                click.echo(click.style(f"✗ Failed: {e}", fg="red"))
        else:
            click.echo(click.style("✓ Server configured", fg="green"))

    asyncio.run(run_test())


def main() -> None:
    """Main entry point for CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
