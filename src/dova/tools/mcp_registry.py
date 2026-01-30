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
        """Invoke MCP tool via stdio transport."""
        # Note: In production, this would use the MCP SDK's stdio client
        # For now, we simulate the invocation

        self._logger.debug(
            "stdio_invoke",
            server=server_config.name,
            tool=tool,
            command=server_config.command,
        )

        # Placeholder - actual implementation would:
        # 1. Start subprocess with server_config.command
        # 2. Send JSON-RPC request over stdin
        # 3. Read response from stdout
        # 4. Parse and return result

        raise NotImplementedError(
            f"STDIO transport for {server_config.name} requires MCP SDK integration"
        )

    async def _invoke_http(
        self,
        server_config: MCPServerConfig,
        tool: str,
        params: dict[str, Any],
    ) -> Any:
        """Invoke MCP tool via HTTP transport."""
        import httpx

        if not server_config.url:
            raise ValueError(f"No URL configured for HTTP server: {server_config.name}")

        async with httpx.AsyncClient(timeout=server_config.timeout_seconds) as client:
            response = await client.post(
                f"{server_config.url}/tools/{tool}",
                json=params,
            )
            response.raise_for_status()
            return response.json()

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
