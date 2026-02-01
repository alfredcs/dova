"""
Model and capability auto-discovery service.

Discovers available LLM models and MCP servers, caches capabilities,
and provides runtime information for intelligent routing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from dova.utils.cache import Cache

logger = structlog.get_logger(__name__)


@dataclass
class ModelInfo:
    """Information about an available LLM model."""

    provider: str
    model_id: str
    context_length: int
    capabilities: list[str] = field(default_factory=list)
    pricing: dict[str, float] = field(default_factory=dict)
    available: bool = True
    last_checked: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "context_length": self.context_length,
            "capabilities": self.capabilities,
            "pricing": self.pricing,
            "available": self.available,
            "last_checked": self.last_checked.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelInfo":
        """Deserialize from dictionary."""
        return cls(
            provider=data["provider"],
            model_id=data["model_id"],
            context_length=data["context_length"],
            capabilities=data.get("capabilities", []),
            pricing=data.get("pricing", {}),
            available=data.get("available", True),
            last_checked=datetime.fromisoformat(data["last_checked"]),
        )


@dataclass
class MCPServerInfo:
    """Information about an MCP server."""

    name: str
    url: str | None = None
    tools: list[str] = field(default_factory=list)
    healthy: bool = True
    last_checked: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "url": self.url,
            "tools": self.tools,
            "healthy": self.healthy,
            "last_checked": self.last_checked.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPServerInfo":
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            url=data.get("url"),
            tools=data.get("tools", []),
            healthy=data.get("healthy", True),
            last_checked=datetime.fromisoformat(data["last_checked"]),
            metadata=data.get("metadata", {}),
        )


# Known model configurations for common providers
KNOWN_MODELS: dict[str, dict[str, Any]] = {
    "anthropic.claude-sonnet-4-20250514-v1:0": {
        "provider": "bedrock",
        "context_length": 200000,
        "capabilities": ["reasoning", "code", "analysis", "vision"],
        "pricing": {"input": 0.003, "output": 0.015},
    },
    "anthropic.claude-3-5-haiku-20241022-v1:0": {
        "provider": "bedrock",
        "context_length": 200000,
        "capabilities": ["reasoning", "code", "fast"],
        "pricing": {"input": 0.0008, "output": 0.004},
    },
    "claude-sonnet-4-20250514": {
        "provider": "anthropic",
        "context_length": 200000,
        "capabilities": ["reasoning", "code", "analysis", "vision"],
        "pricing": {"input": 0.003, "output": 0.015},
    },
    "gpt-4o": {
        "provider": "openai",
        "context_length": 128000,
        "capabilities": ["reasoning", "code", "analysis", "vision"],
        "pricing": {"input": 0.005, "output": 0.015},
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "context_length": 128000,
        "capabilities": ["reasoning", "code", "fast"],
        "pricing": {"input": 0.00015, "output": 0.0006},
    },
}


class AutoDiscovery:
    """
    Auto-discovers and caches information about available models and MCP servers.

    Provides a unified view of capabilities across providers for intelligent
    model selection and fallback routing.
    """

    def __init__(
        self,
        cache: Cache,
        llm_router: Any | None = None,
        mcp_client: Any | None = None,
        cache_ttl: int = 3600,
    ):
        self.cache = cache
        self.llm_router = llm_router
        self.mcp_client = mcp_client
        self.cache_ttl = cache_ttl
        self._logger = logger.bind(service="discovery")

    async def discover_models(self, force_refresh: bool = False) -> list[ModelInfo]:
        """
        Discover available LLM models.

        Args:
            force_refresh: Skip cache and rediscover

        Returns:
            List of discovered models
        """
        cache_key = "discovery:models"

        # Check cache first
        if not force_refresh:
            cached = await self.cache.get(cache_key)
            if cached:
                return [ModelInfo.from_dict(m) for m in cached]

        models: list[ModelInfo] = []

        # Discover from LLM router providers
        if self.llm_router:
            for name, provider in self.llm_router.providers.items():
                for task_type, model_config in provider.config.models.items():
                    model_id = model_config.model_id

                    # Get known info or create basic info
                    known = KNOWN_MODELS.get(model_id, {})
                    model_info = ModelInfo(
                        provider=name,
                        model_id=model_id,
                        context_length=known.get("context_length", 4096),
                        capabilities=known.get("capabilities", [task_type.value]),
                        pricing=known.get("pricing", {}),
                        available=provider.config.enabled,
                    )

                    # Avoid duplicates
                    if not any(m.model_id == model_id for m in models):
                        models.append(model_info)

        # Health check models
        for model in models:
            model.available = await self._check_model_health(model)

        # Cache results
        await self.cache.set(
            cache_key,
            [m.to_dict() for m in models],
            ttl=self.cache_ttl,
        )

        self._logger.info("models_discovered", count=len(models))
        return models

    async def discover_mcp_servers(
        self,
        force_refresh: bool = False,
    ) -> list[MCPServerInfo]:
        """
        Discover available MCP servers.

        Args:
            force_refresh: Skip cache and rediscover

        Returns:
            List of discovered MCP servers
        """
        cache_key = "discovery:mcp_servers"

        # Check cache first
        if not force_refresh:
            cached = await self.cache.get(cache_key)
            if cached:
                return [MCPServerInfo.from_dict(s) for s in cached]

        servers: list[MCPServerInfo] = []

        # Discover from MCP client if available
        if self.mcp_client:
            try:
                # This depends on your MCP client implementation
                if hasattr(self.mcp_client, "list_servers"):
                    raw_servers = await self.mcp_client.list_servers()
                    for server_data in raw_servers:
                        server_info = MCPServerInfo(
                            name=server_data.get("name", "unknown"),
                            url=server_data.get("url"),
                            tools=server_data.get("tools", []),
                        )
                        servers.append(server_info)
            except Exception as e:
                self._logger.warning("mcp_discovery_failed", error=str(e))

        # Add known default servers if none discovered
        if not servers:
            servers = self._get_default_mcp_servers()

        # Health check servers
        for server in servers:
            server.healthy = await self._check_mcp_health(server)

        # Cache results
        await self.cache.set(
            cache_key,
            [s.to_dict() for s in servers],
            ttl=self.cache_ttl,
        )

        self._logger.info("mcp_servers_discovered", count=len(servers))
        return servers

    async def refresh_all(self) -> dict[str, Any]:
        """
        Refresh all discovery caches.

        Returns:
            Summary of discovered resources
        """
        models = await self.discover_models(force_refresh=True)
        mcp_servers = await self.discover_mcp_servers(force_refresh=True)

        summary = {
            "models": {
                "total": len(models),
                "available": sum(1 for m in models if m.available),
                "providers": list(set(m.provider for m in models)),
            },
            "mcp_servers": {
                "total": len(mcp_servers),
                "healthy": sum(1 for s in mcp_servers if s.healthy),
                "names": [s.name for s in mcp_servers],
            },
            "refreshed_at": datetime.utcnow().isoformat(),
        }

        self._logger.info("discovery_refresh_complete", summary=summary)
        return summary

    async def get_model_by_capability(
        self,
        capability: str,
        prefer_provider: str | None = None,
    ) -> ModelInfo | None:
        """
        Find a model with a specific capability.

        Args:
            capability: Required capability (e.g., "vision", "code")
            prefer_provider: Preferred provider

        Returns:
            Matching model or None
        """
        models = await self.discover_models()

        # Filter by capability and availability
        matching = [
            m for m in models
            if capability in m.capabilities and m.available
        ]

        if not matching:
            return None

        # Prefer specific provider if requested
        if prefer_provider:
            provider_match = [m for m in matching if m.provider == prefer_provider]
            if provider_match:
                return provider_match[0]

        return matching[0]

    async def get_mcp_server_for_tool(self, tool_name: str) -> MCPServerInfo | None:
        """
        Find an MCP server that provides a specific tool.

        Args:
            tool_name: Tool to look for

        Returns:
            Server info or None
        """
        servers = await self.discover_mcp_servers()

        for server in servers:
            if tool_name in server.tools and server.healthy:
                return server

        return None

    async def _check_model_health(self, model: ModelInfo) -> bool:
        """Check if a model is available."""
        if not self.llm_router:
            return True

        provider = self.llm_router.providers.get(model.provider)
        if provider is None:
            return False

        try:
            return await provider.health_check()
        except Exception:
            return False

    async def _check_mcp_health(self, server: MCPServerInfo) -> bool:
        """Check if an MCP server is healthy."""
        # Simple check - in production you'd ping the server
        return True

    def _get_default_mcp_servers(self) -> list[MCPServerInfo]:
        """Get default MCP server configurations."""
        return [
            MCPServerInfo(
                name="arxiv",
                tools=["search_arxiv", "get_paper"],
                metadata={"category": "research"},
            ),
            MCPServerInfo(
                name="github",
                tools=["search_repos", "get_repo", "list_issues"],
                metadata={"category": "code"},
            ),
            MCPServerInfo(
                name="huggingface",
                tools=["search_models", "get_model", "search_datasets"],
                metadata={"category": "ml"},
            ),
        ]
