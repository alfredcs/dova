"""AgentCore Runtime support for DOVA.

Provides the BedrockAgentCoreApp instance and entrypoint for
AWS AgentCore Runtime deployment.
"""

import os
from collections.abc import AsyncIterator
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Lazy import to avoid errors when bedrock-agentcore is not installed
_app = None


def get_app() -> Any:
    """Get or create the BedrockAgentCoreApp instance.

    Returns:
        BedrockAgentCoreApp instance

    Raises:
        ImportError: If bedrock-agentcore is not installed
    """
    global _app
    if _app is None:
        try:
            from bedrock_agentcore.runtime import BedrockAgentCoreApp

            _app = BedrockAgentCoreApp()
        except ImportError as e:
            raise ImportError(
                "bedrock-agentcore package is required for AgentCore runtime. "
                "Install with: pip install 'bedrock-agentcore[strands-agents]>=1.0.6'"
            ) from e
    return _app


# Create app instance (lazy)
class LazyApp:
    """Lazy wrapper for BedrockAgentCoreApp to avoid import errors."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_app(), name)

    def run(self, port: int = 8080, host: str | None = None, **kwargs) -> None:
        """Run the AgentCore app.

        Args:
            port: Port to serve on, defaults to 8080
            host: Host to bind to, auto-detected if None
            **kwargs: Additional arguments passed to uvicorn.run()
        """
        get_app().run(port=port, host=host, **kwargs)

    def entrypoint(self, func: Any) -> Any:
        """Decorator for entrypoint functions."""
        return get_app().entrypoint(func)


app = LazyApp()


async def create_agent_with_memory(
    user_id: str,  # noqa: ARG001 - Reserved for future memory integration
    session_id: str,  # noqa: ARG001 - Reserved for future memory integration
) -> Any:
    """Create a DOVA agent with AgentCore Memory integration.

    Args:
        user_id: User identifier for memory namespace
        session_id: Session identifier

    Returns:
        Configured agent instance
    """
    from dova.agents.orchestrator import DOVAOrchestrator
    from dova.agents.research import ResearchAgent
    from dova.agents.synthesis import SynthesisAgent
    from dova.config.providers import create_llm_router_from_settings
    from dova.config.settings import get_settings
    from dova.tools.mcp_registry import MCPClient, create_gateway_mcp_client

    settings = get_settings()
    llm_router = create_llm_router_from_settings()

    # Try to create gateway MCP client for AgentCore
    gateway_config = create_gateway_mcp_client()

    # Create MCP client with gateway support
    mcp_client = MCPClient()
    if gateway_config:
        mcp_client.registry.register_server(gateway_config)

    # Create agents
    research_agent = ResearchAgent(
        llm_router=llm_router,
        mcp_client=mcp_client,
        tavily_api_key=settings.mcp.tavily_api_key,
    )
    synthesis_agent = SynthesisAgent(llm_router=llm_router)

    orchestrator = DOVAOrchestrator(
        llm_router=llm_router,
        mcp_client=mcp_client,
        agents={
            "research": research_agent,
            "synthesis": synthesis_agent,
        },
    )

    return orchestrator


async def agent_stream(payload: dict[str, Any]) -> AsyncIterator[str]:
    """Main entrypoint for AgentCore Runtime.

    This function is called by the AgentCore Runtime with the request payload.
    It creates an agent, processes the request, and streams the response.

    Args:
        payload: Request payload containing:
            - prompt: User query
            - userId: User identifier
            - runtimeSessionId: Session identifier
            - Additional context parameters

    Yields:
        Response chunks as strings
    """
    # Extract payload fields
    user_query = payload.get("prompt", "")
    user_id = payload.get("userId", "anonymous")
    session_id = payload.get("runtimeSessionId", "")

    logger.info(
        "agentcore_request",
        user_id=user_id,
        session_id=session_id[:16] + "..." if len(session_id) > 16 else session_id,
        query_len=len(user_query),
    )

    try:
        # Create agent with memory integration
        orchestrator = await create_agent_with_memory(user_id, session_id)

        # Build task
        from dova.agents.base import AgentTask

        task = AgentTask(
            type="research",
            params={
                "query": user_query,
                "sources": ["github", "huggingface", "web"],
                "reasoning_mode": "standard",
            },
            user_id=user_id,
        )

        # Execute and stream response
        result = await orchestrator.execute(task)

        if result.success:
            # Format response
            response = format_response(result.data)
            yield response
        else:
            yield f"Error: {result.error}"

    except Exception as e:
        logger.exception("agentcore_error", error=str(e))
        yield f"An error occurred: {str(e)}"


def format_response(data: dict[str, Any] | None) -> str:
    """Format agent result data for response.

    Args:
        data: Agent result data

    Returns:
        Formatted response string
    """
    if not data:
        return "No results found."

    parts = []

    if "summary" in data and data["summary"]:
        summary = data["summary"]
        if isinstance(summary, dict):
            summary = summary.get("text", summary.get("content", str(summary)))
        parts.append(f"## Summary\n{summary}")

    if "insights" in data and data["insights"]:
        parts.append("\n## Key Findings")
        for insight in data["insights"][:5]:
            if isinstance(insight, dict):
                parts.append(f"- {insight.get('title', insight.get('summary', str(insight)))}")
            else:
                parts.append(f"- {insight}")

    if "recommendations" in data and data["recommendations"]:
        parts.append("\n## Recommendations")
        for rec in data["recommendations"][:3]:
            if isinstance(rec, dict):
                action = rec.get("action", rec.get("recommendation", str(rec)))
                parts.append(f"- {action}")
            else:
                parts.append(f"- {rec}")

    return "\n".join(parts) if parts else "Research completed."


# Register entrypoint with decorator when module is loaded in AgentCore mode
def _register_entrypoint() -> None:
    """Register the agent_stream function as the AgentCore entrypoint."""
    if os.environ.get("RUNTIME_MODE") == "agentcore":
        try:
            real_app = get_app()

            @real_app.entrypoint
            async def _entrypoint(payload: dict[str, Any]) -> AsyncIterator[str]:
                async for chunk in agent_stream(payload):
                    yield chunk

        except ImportError:
            pass  # Not in AgentCore environment


# Attempt registration on import
_register_entrypoint()
