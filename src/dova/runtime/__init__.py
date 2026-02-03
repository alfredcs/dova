"""DOVA Runtime Module.

Provides runtime support for different deployment modes:
- FastAPI (local development, full flexibility)
- AgentCore Runtime (AWS deployment with @app.entrypoint pattern)
"""

from dova.runtime.agentcore_app import agent_stream, app

__all__ = ["app", "agent_stream"]
