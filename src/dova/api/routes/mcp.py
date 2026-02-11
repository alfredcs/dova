"""MCP server management API endpoints."""
from datetime import datetime

import httpx
import structlog
from fastapi import APIRouter

from dova.api.schemas.mcp import MCPServerListResponse, MCPServerSchema
from dova.config.mcp_servers import (
    get_default_registry,
    list_mcp_servers,
    load_managed_mcp_servers,
)

router = APIRouter(prefix="/mcp")
logger = structlog.get_logger(__name__)


async def check_http_server(url: str, headers: dict) -> tuple[bool, str]:
    """Check if an HTTP MCP server is reachable."""
    request_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    request_headers.update(headers)

    try:
        async with httpx.AsyncClient(timeout=10) as client:
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
                headers=request_headers,
            )
            if response.status_code == 200:
                return True, "OK"
            return False, f"HTTP {response.status_code}"
    except httpx.ConnectError:
        return False, "Connection refused"
    except httpx.TimeoutException:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)[:50]


@router.get("/servers", response_model=MCPServerListResponse)
async def list_servers(check_health: bool = False) -> MCPServerListResponse:
    """List all configured MCP servers.

    Args:
        check_health: If true, check connectivity to HTTP servers.

    Returns:
        List of MCP servers with their status.
    """
    servers_list = []

    # Get servers from config file
    config_servers = list_mcp_servers()

    # Get managed repo servers
    managed_servers = load_managed_mcp_servers()

    # Merge managed servers (don't override config)
    all_servers = dict(config_servers)
    for name, config in managed_servers.items():
        if name not in all_servers:
            all_servers[name] = {
                "type": "stdio",
                "command": config.command,
                "managed": True,
            }

    for name, config in all_servers.items():
        # Detect server type
        is_stdio = config.get("type") == "stdio" or (
            config.get("command") and not config.get("url")
        )
        transport = "stdio" if is_stdio else config.get("type", "http")

        # Build command string for stdio servers
        command = None
        if is_stdio:
            cmd = config.get("command", "")
            args = config.get("args", [])
            if args:
                command = f"{cmd} {' '.join(args)}"
            else:
                command = cmd

        # Determine status
        status = "unknown"
        status_message = None

        if check_health:
            if is_stdio:
                status = "healthy"
                status_message = "STDIO (local)"
            else:
                url = config.get("url")
                if url:
                    headers = config.get("headers", {})
                    ok, msg = await check_http_server(url, headers)
                    status = "healthy" if ok else "unhealthy"
                    status_message = msg
                else:
                    status = "unhealthy"
                    status_message = "No URL configured"

        # Get description from registry if available
        registry = get_default_registry()
        registry_server = registry.get_server(name)
        description = (
            registry_server.description
            if registry_server
            else f"MCP server: {name}"
        )

        servers_list.append(
            MCPServerSchema(
                name=name,
                description=description,
                transport=transport,
                enabled=config.get("enabled", True),
                url=config.get("url") if not is_stdio else None,
                command=command,
                status=status,
                status_message=status_message,
            )
        )

    # Sort: HTTP servers first, then alphabetically
    servers_list.sort(key=lambda s: (s.transport != "http", s.name))

    return MCPServerListResponse(
        servers=servers_list,
        total=len(servers_list),
        timestamp=datetime.utcnow(),
    )
