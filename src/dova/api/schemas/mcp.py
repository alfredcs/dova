"""MCP server schemas."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class MCPServerSchema(BaseModel):
    """Schema for an MCP server."""

    name: str
    description: Optional[str] = None
    transport: str  # "stdio", "http", "sse"
    enabled: bool = True
    url: Optional[str] = None
    command: Optional[str] = None
    status: str = "unknown"  # "healthy", "unhealthy", "unknown"
    status_message: Optional[str] = None


class MCPServerListResponse(BaseModel):
    """Response for listing MCP servers."""

    servers: list[MCPServerSchema]
    total: int
    timestamp: datetime
