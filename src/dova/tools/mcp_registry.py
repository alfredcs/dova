"""
MCP Client and Server Registry for DOVA.

Manages connections to MCP servers and provides a unified
interface for tool invocations.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import structlog

from dova.config.mcp_servers import (
    MCPRegistry,
    MCPServerConfig,
    MCPTransport,
    get_default_registry,
)
from dova.utils.cache import Cache, InMemoryCache
from dova.utils.metrics import MetricsCollector, MetricNames
from dova.utils.retry import CircuitBreaker, RetryConfig, retry_async

logger = structlog.get_logger(__name__)


def create_gateway_mcp_client(access_token: str | None = None) -> MCPServerConfig | None:
    """Create MCP server config for AgentCore Gateway.

    Returns None if gateway is not configured (local development).
    Falls back to existing MCP servers when gateway unavailable.

    Args:
        access_token: Optional pre-fetched access token

    Returns:
        MCPServerConfig for gateway or None if not configured
    """
    import os

    stack_name = os.environ.get("STACK_NAME")

    if not stack_name:
        logger.warning(
            "gateway_not_configured",
            reason="STACK_NAME not set",
            fallback="Using existing MCP servers (STDIO/HTTP)",
        )
        return None

    try:
        from dova.services.gateway_auth import get_gateway_access_token, get_gateway_url

        if access_token is None:
            access_token = get_gateway_access_token()

        gateway_url = get_gateway_url()

        return MCPServerConfig(
            name="gateway",
            description="AgentCore Gateway MCP Server",
            transport=MCPTransport.STREAMABLE_HTTP,
            url=gateway_url,
            headers={"Authorization": f"Bearer {access_token}"},
            enabled=True,
            priority=0,  # Highest priority
        )
    except Exception as e:
        logger.warning(
            "gateway_client_creation_failed",
            error=str(e),
            fallback="Using existing MCP servers",
        )
        return None


@dataclass
class MCPInvocationResult:
    """Result from an MCP tool invocation."""

    success: bool
    data: Any = None
    error: str | None = None
    server: str = ""
    tool: str = ""
    cached: bool = False
    latency_ms: float = 0.0


class MCPClient:
    """
    Client for interacting with MCP servers.

    Provides a unified interface for invoking tools across
    different MCP servers with retry, caching, and circuit breaker.
    """

    def __init__(
        self,
        registry: MCPRegistry | None = None,
        cache: Cache | None = None,
        metrics: MetricsCollector | None = None,
        retry_config: RetryConfig | None = None,
    ):
        self.registry = registry or get_default_registry()
        self.cache = cache or InMemoryCache()
        self.metrics = metrics or MetricsCollector()
        self.retry_config = retry_config or RetryConfig(max_retries=3)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        self._connections: dict[str, Any] = {}
        self._sessions: dict[str, str] = {}  # server_name -> session_id
        self._logger = logger.bind(component="mcp_client")

    async def invoke(
        self,
        server: str,
        tool: str,
        params: dict[str, Any],
        use_cache: bool = True,
        cache_ttl: float = 3600,
    ) -> Any:
        """
        Invoke an MCP tool.

        Args:
            server: MCP server name
            tool: Tool name to invoke
            params: Tool parameters
            use_cache: Whether to use cached results
            cache_ttl: Cache time-to-live in seconds

        Returns:
            Tool result data

        Raises:
            ValueError: If server or tool not found
            RuntimeError: If invocation fails
        """
        # Validate server exists
        server_config = self.registry.get_server(server)
        if not server_config:
            raise ValueError(f"Unknown MCP server: {server}")

        if not server_config.enabled:
            raise ValueError(f"MCP server is disabled: {server}")

        # Check cache
        cache_key = self._cache_key(server, tool, params)
        if use_cache:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                self._logger.debug("cache_hit", server=server, tool=tool)
                self.metrics.increment(MetricNames.CACHE_HIT_COUNT)
                return cached

        self.metrics.increment(MetricNames.CACHE_MISS_COUNT)

        # Invoke with circuit breaker
        try:
            result = await self.circuit_breaker.call(
                f"{server}:{tool}",
                self._invoke_internal,
                server_config,
                tool,
                params,
            )

            # Cache successful result
            if use_cache:
                await self.cache.set(cache_key, result, ttl=cache_ttl)

            return result

        except Exception as e:
            self._logger.error(
                "mcp_invoke_failed",
                server=server,
                tool=tool,
                error=str(e),
            )
            raise

    async def _invoke_internal(
        self,
        server_config: MCPServerConfig,
        tool: str,
        params: dict[str, Any],
    ) -> Any:
        """Internal invocation with retry logic."""
        import time

        async def _call() -> Any:
            start_time = time.time()

            if server_config.transport == MCPTransport.STDIO:
                result = await self._invoke_stdio(server_config, tool, params)
            elif server_config.transport == MCPTransport.HTTP:
                result = await self._invoke_http(server_config, tool, params)
            elif server_config.transport == MCPTransport.SSE:
                result = await self._invoke_sse(server_config, tool, params)
            elif server_config.transport == MCPTransport.STREAMABLE_HTTP:
                result = await self._invoke_streamable_http(server_config, tool, params)
            else:
                raise ValueError(f"Unsupported transport: {server_config.transport}")

            latency_ms = (time.time() - start_time) * 1000
            self.metrics.record(
                MetricNames.MCP_CALL_LATENCY,
                latency_ms,
                unit="ms",
                labels={"server": server_config.name, "tool": tool},
            )
            self.metrics.increment(
                MetricNames.MCP_CALL_COUNT,
                labels={"server": server_config.name, "tool": tool, "status": "success"},
            )

            return result

        return await retry_async(_call, config=self.retry_config)

    async def _invoke_stdio(
        self,
        server_config: MCPServerConfig,
        tool: str,
        params: dict[str, Any],
    ) -> Any:
        """Invoke MCP tool via stdio transport with proper initialization."""
        import json
        import os

        self._logger.debug(
            "stdio_invoke",
            server=server_config.name,
            tool=tool,
            command=server_config.command,
        )

        if not server_config.command:
            raise ValueError(f"No command configured for STDIO server: {server_config.name}")

        # Get command from repo manager if available
        from dova.services.mcp_repo_manager import get_mcp_repo_manager

        repo_manager = get_mcp_repo_manager()
        cmd_info = repo_manager.get_server_command(server_config.name)

        if cmd_info:
            command, args, env_vars = cmd_info
            full_command = [command] + args
            env = {**os.environ, **env_vars}
        else:
            # Fallback to configured command
            full_command = server_config.command.split()
            env = {**os.environ, **server_config.env_vars}

        # Start process
        # `limit` bumps asyncio StreamReader's per-line buffer from 64 KiB to
        # 10 MiB. Arxiv `search_papers` with ~10 full abstracts routinely
        # exceeds 64 KiB in one JSON-RPC line, tripping
        # "Separator is not found, and chunk exceed the limit" on
        # process.stdout.readline(). 10 MiB covers any realistic MCP payload
        # without being so high we mask a runaway response.
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *full_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                limit=10 * 1024 * 1024,
            )

            async def send_and_receive(request: dict, request_id: int) -> dict:
                """Send a JSON-RPC request and read the response."""
                request["id"] = request_id
                request_line = json.dumps(request) + "\n"
                process.stdin.write(request_line.encode())
                await process.stdin.drain()

                # Read response line
                response_line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=server_config.timeout_seconds,
                )
                if not response_line:
                    raise RuntimeError("No response from server")
                return json.loads(response_line.decode().strip())

            # Step 1: Initialize
            init_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "dova", "version": "1.1.0"},
                },
            }
            init_response = await send_and_receive(init_request, 0)
            if "error" in init_response:
                raise RuntimeError(f"Initialize failed: {init_response['error']}")

            # Step 2: Send initialized notification
            initialized_notif = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
            process.stdin.write((json.dumps(initialized_notif) + "\n").encode())
            await process.stdin.drain()

            # Step 3: Call the tool
            tool_request = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": tool, "arguments": params},
            }
            tool_response = await send_and_receive(tool_request, 1)

            # Cleanup
            process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                process.kill()

            return self._parse_stdio_result(tool_response)

        except asyncio.TimeoutError:
            self._logger.error(
                "stdio_timeout",
                server=server_config.name,
                timeout=server_config.timeout_seconds,
            )
            if process:
                process.kill()
            raise RuntimeError(f"STDIO request timed out after {server_config.timeout_seconds}s")
        except Exception as e:
            if process and process.returncode is None:
                process.kill()
            raise

    def _parse_stdio_result(self, response: dict[str, Any]) -> Any:
        """Parse JSON-RPC response from STDIO server."""
        import json

        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")

        result = response.get("result", {})
        content = result.get("content", [])

        if content and isinstance(content, list):
            texts = [c.get("text", "") for c in content if c.get("type") == "text"]
            if texts:
                combined = "\n".join(texts)
                try:
                    return json.loads(combined)
                except json.JSONDecodeError:
                    return combined

        return result

    async def _invoke_http(
        self,
        server_config: MCPServerConfig,
        tool: str,
        params: dict[str, Any],
    ) -> Any:
        """Invoke MCP tool via HTTP transport with session management."""
        import httpx
        import json

        if not server_config.url:
            raise ValueError(f"No URL configured for HTTP server: {server_config.name}")

        async with httpx.AsyncClient(timeout=server_config.timeout_seconds) as client:
            # Ensure we have a session for this server
            if server_config.name not in self._sessions:
                await self._initialize_session(client, server_config)

            # Build headers with session ID
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            }
            headers.update(server_config.headers)

            # Add session ID if we have one
            session_id = self._sessions.get(server_config.name)
            if session_id:
                headers["Mcp-Session-Id"] = session_id

            self._logger.debug(
                "http_invoke",
                server=server_config.name,
                tool=tool,
                url=server_config.url,
            )

            # MCP HTTP uses JSON-RPC 2.0
            request_body = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool, "arguments": params},
            }

            response = await client.post(
                server_config.url,
                json=request_body,
                headers=headers,
            )

            # Update session ID from response if provided
            if "mcp-session-id" in response.headers:
                self._sessions[server_config.name] = response.headers["mcp-session-id"]

            # Check for HTTP errors with detailed message
            if response.status_code >= 400:
                self._logger.error(
                    "http_error",
                    server=server_config.name,
                    status=response.status_code,
                    body=response.text[:500] if response.text else "(empty)",
                )
                response.raise_for_status()

            return self._parse_http_response(server_config.name, response)

    async def _initialize_session(
        self,
        client: Any,
        server_config: MCPServerConfig,
    ) -> None:
        """Initialize an MCP session with the server."""
        import json

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        headers.update(server_config.headers)

        self._logger.debug(
            "session_init",
            server=server_config.name,
            url=server_config.url,
        )

        # Send initialize request per MCP spec
        init_body = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "dova",
                    "version": "1.1.0",
                },
            },
        }

        response = await client.post(
            server_config.url,
            json=init_body,
            headers=headers,
        )

        # Capture session ID from response header
        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self._sessions[server_config.name] = session_id
            self._logger.debug(
                "session_created",
                server=server_config.name,
                session_id=session_id[:16] + "..." if len(session_id) > 16 else session_id,
            )
        else:
            # Some servers might not require session IDs
            self._logger.debug(
                "session_no_id",
                server=server_config.name,
            )

        if response.status_code >= 400:
            self._logger.error(
                "session_init_error",
                server=server_config.name,
                status=response.status_code,
                body=response.text[:500] if response.text else "(empty)",
            )
            response.raise_for_status()

    def _parse_http_response(self, server_name: str, response: Any) -> Any:
        """Parse HTTP response from MCP server."""
        import json

        # Handle empty response
        if not response.text or not response.text.strip():
            raise RuntimeError(
                f"Empty response from {server_name}. "
                f"Status: {response.status_code}. "
                "Check server URL and authentication."
            )

        # Try to parse response - handle both JSON and SSE formats
        response_text = response.text.strip()
        result = None

        # Check if response is SSE format (starts with "event:" or "data:")
        if response_text.startswith("event:") or response_text.startswith("data:"):
            # Parse SSE format - extract JSON from data lines
            for line in response_text.split("\n"):
                line = line.strip()
                if line.startswith("data:"):
                    json_str = line[5:].strip()
                    if json_str:
                        try:
                            result = json.loads(json_str)
                            break
                        except json.JSONDecodeError:
                            continue

            if result is None:
                self._logger.error(
                    "sse_parse_error",
                    server=server_name,
                    body=response_text[:500],
                )
                raise RuntimeError(f"Could not parse SSE response from {server_name}")
        else:
            # Regular JSON response
            try:
                result = response.json()
            except Exception as e:
                self._logger.error(
                    "json_parse_error",
                    server=server_name,
                    body=response_text[:500],
                )
                raise RuntimeError(f"Invalid JSON from {server_name}: {e}")

        # Handle JSON-RPC response
        if "error" in result:
            raise RuntimeError(f"MCP error: {result['error']}")

        # Extract content from result
        mcp_result = result.get("result", {})
        content = mcp_result.get("content", [])

        if content and isinstance(content, list):
            texts = [c.get("text", "") for c in content if c.get("type") == "text"]
            if texts:
                combined = "\n".join(texts)
                try:
                    return json.loads(combined)
                except json.JSONDecodeError:
                    return combined

        return mcp_result

    async def _invoke_sse(
        self,
        server_config: MCPServerConfig,
        tool: str,
        params: dict[str, Any],
    ) -> Any:
        """Invoke MCP tool via SSE transport."""
        # SSE is typically used for streaming responses
        # For tool invocations, we'd use the request-response pattern
        raise NotImplementedError(
            f"SSE transport for {server_config.name} not yet implemented"
        )

    async def _invoke_streamable_http(
        self,
        server_config: MCPServerConfig,
        tool: str,
        params: dict[str, Any],
    ) -> Any:
        """Invoke MCP tool via Streamable HTTP transport (AgentCore Gateway).

        Uses the MCP Streamable HTTP transport for communication with
        AgentCore Gateway endpoints.
        """
        try:
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError:
            raise ImportError(
                "mcp package with streamable_http support is required. "
                "Install with: pip install 'mcp>=1.21.0'"
            )

        if not server_config.url:
            raise ValueError(
                f"No URL configured for Streamable HTTP server: {server_config.name}"
            )

        self._logger.debug(
            "streamable_http_invoke",
            server=server_config.name,
            tool=tool,
            url=server_config.url,
        )

        async with streamablehttp_client(
            url=server_config.url,
            headers=server_config.headers,
        ) as (read_stream, write_stream, _):
            # Import MCP client session
            from mcp import ClientSession

            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Call the tool
                result = await session.call_tool(tool, params)

                return self._parse_streamable_result(result)

    def _parse_streamable_result(self, result: Any) -> Any:
        """Parse result from streamable HTTP client."""
        import json

        # Handle MCP CallToolResult
        if hasattr(result, "content"):
            content = result.content
            if content and isinstance(content, list):
                texts = []
                for item in content:
                    if hasattr(item, "text"):
                        texts.append(item.text)
                    elif isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))
                if texts:
                    combined = "\n".join(texts)
                    try:
                        return json.loads(combined)
                    except json.JSONDecodeError:
                        return combined
            return content

        return result

    def _cache_key(self, server: str, tool: str, params: dict[str, Any]) -> str:
        """Generate cache key for MCP invocation."""
        import hashlib
        import json

        param_str = json.dumps(params, sort_keys=True)
        key_str = f"{server}:{tool}:{param_str}"
        return f"mcp:{hashlib.sha256(key_str.encode()).hexdigest()[:16]}"

    async def health_check(self, server: str | None = None) -> dict[str, bool]:
        """Check health of MCP servers."""
        servers = [server] if server else list(self.registry.servers.keys())
        results = {}

        for server_name in servers:
            server_config = self.registry.get_server(server_name)
            if not server_config or not server_config.enabled:
                results[server_name] = False
                continue

            try:
                # Simple health check - varies by transport
                results[server_name] = True  # Placeholder
            except Exception:
                results[server_name] = False

        return results


class MCPManager:
    """
    Manager for MCP server lifecycle and connections.

    Handles starting/stopping MCP servers and maintaining connections.
    """

    def __init__(
        self,
        registry: MCPRegistry | None = None,
        metrics: MetricsCollector | None = None,
    ):
        self.registry = registry or get_default_registry()
        self.metrics = metrics or MetricsCollector()
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._clients: dict[str, MCPClient] = {}
        self._logger = logger.bind(component="mcp_manager")

    async def start_server(self, server_name: str) -> bool:
        """Start an MCP server."""
        server_config = self.registry.get_server(server_name)
        if not server_config:
            self._logger.error("server_not_found", server=server_name)
            return False

        if not server_config.enabled:
            self._logger.warning("server_disabled", server=server_name)
            return False

        if server_config.transport != MCPTransport.STDIO:
            self._logger.info(
                "non_stdio_server",
                server=server_name,
                transport=server_config.transport.value,
            )
            return True  # Non-stdio servers don't need to be started

        try:
            if server_config.command:
                # Start subprocess
                process = await asyncio.create_subprocess_shell(
                    server_config.command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                self._processes[server_name] = process
                self._logger.info("server_started", server=server_name, pid=process.pid)
                return True

        except Exception as e:
            self._logger.error("server_start_failed", server=server_name, error=str(e))

        return False

    async def stop_server(self, server_name: str) -> bool:
        """Stop an MCP server."""
        if server_name not in self._processes:
            return True

        process = self._processes[server_name]
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=5.0)
            self._logger.info("server_stopped", server=server_name)
        except asyncio.TimeoutError:
            process.kill()
            self._logger.warning("server_killed", server=server_name)
        finally:
            del self._processes[server_name]

        return True

    async def start_all(self) -> dict[str, bool]:
        """Start all enabled MCP servers."""
        results = {}
        for server_name in self.registry.servers:
            if self.registry.servers[server_name].enabled:
                results[server_name] = await self.start_server(server_name)
        return results

    async def stop_all(self) -> None:
        """Stop all running MCP servers."""
        for server_name in list(self._processes.keys()):
            await self.stop_server(server_name)

    def get_client(self, cache: Cache | None = None) -> MCPClient:
        """Get or create MCP client."""
        key = "default"
        if key not in self._clients:
            self._clients[key] = MCPClient(
                registry=self.registry,
                cache=cache,
                metrics=self.metrics,
            )
        return self._clients[key]

    async def __aenter__(self) -> "MCPManager":
        """Async context manager entry."""
        await self.start_all()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop_all()
