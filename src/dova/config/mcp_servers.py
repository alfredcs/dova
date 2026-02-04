"""
MCP Server Configuration and Registry.

Defines available MCP servers and their capabilities for the DOVA platform.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MCPTransport(Enum):
    """MCP transport protocols."""

    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"
    STREAMABLE_HTTP = "streamable_http"  # AgentCore Gateway


class MCPCapability(Enum):
    """MCP server capabilities."""

    SEARCH = "search"
    FETCH = "fetch"
    LIST = "list"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


@dataclass
class MCPTool:
    """Definition of an MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    capabilities: list[MCPCapability] = field(default_factory=list)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""

    name: str
    description: str
    transport: MCPTransport
    command: str | None = None  # For stdio transport
    url: str | None = None  # For http/sse transport
    headers: dict[str, str] = field(default_factory=dict)  # For http transport
    enabled: bool = True
    priority: int = 5  # 1=highest, 10=lowest
    tools: list[MCPTool] = field(default_factory=list)
    rate_limit_rpm: int = 60
    timeout_seconds: int = 30
    env_vars: dict[str, str] = field(default_factory=dict)


# ArXiv MCP Server
ARXIV_MCP = MCPServerConfig(
    name="arxiv",
    description="ArXiv paper search and metadata retrieval",
    transport=MCPTransport.STDIO,
    command="uvx mcp-server-arxiv",
    priority=1,
    tools=[
        MCPTool(
            name="search_arxiv",  # Note: actual MCP tool name
            description="Search ArXiv papers by query string",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {
                        "type": "integer",
                        "default": 10,
                        "maximum": 100,
                    },
                    "start": {"type": "integer", "default": 0},
                },
                "required": ["query"],
            },
            capabilities=[MCPCapability.SEARCH],
        ),
        MCPTool(
            name="search_by_author",
            description="Search ArXiv papers by author name",
            input_schema={
                "type": "object",
                "properties": {
                    "author": {"type": "string", "description": "Author name"},
                    "max_results": {"type": "integer", "default": 10},
                },
                "required": ["author"],
            },
            capabilities=[MCPCapability.SEARCH],
        ),
        MCPTool(
            name="search_by_category",
            description="Search ArXiv papers by category",
            input_schema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "ArXiv category (e.g., cs.AI, cs.LG)",
                    },
                    "max_results": {"type": "integer", "default": 10},
                },
                "required": ["category"],
            },
            capabilities=[MCPCapability.SEARCH],
        ),
    ],
    rate_limit_rpm=30,
    timeout_seconds=60,
)


# GitHub MCP Server
GITHUB_MCP = MCPServerConfig(
    name="github",
    description="GitHub repository search, code search, and API access",
    transport=MCPTransport.STDIO,
    command="npx -y @modelcontextprotocol/server-github",
    priority=1,
    tools=[
        MCPTool(
            name="search_repositories",
            description="Search GitHub repositories",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "sort": {
                        "type": "string",
                        "enum": ["stars", "forks", "updated"],
                    },
                    "order": {"type": "string", "enum": ["asc", "desc"]},
                    "per_page": {"type": "integer", "default": 30},
                },
                "required": ["query"],
            },
            capabilities=[MCPCapability.SEARCH],
        ),
        MCPTool(
            name="search_code",
            description="Search code across GitHub repositories",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "per_page": {"type": "integer", "default": 30},
                },
                "required": ["query"],
            },
            capabilities=[MCPCapability.SEARCH],
        ),
        MCPTool(
            name="get_file_contents",
            description="Get contents of a file from a repository",
            input_schema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "path": {"type": "string"},
                    "ref": {"type": "string"},
                },
                "required": ["owner", "repo", "path"],
            },
            capabilities=[MCPCapability.FETCH],
        ),
        MCPTool(
            name="list_issues",
            description="List issues in a repository",
            input_schema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"},
                    "state": {"type": "string", "enum": ["open", "closed", "all"]},
                    "per_page": {"type": "integer", "default": 30},
                },
                "required": ["owner", "repo"],
            },
            capabilities=[MCPCapability.LIST],
        ),
    ],
    rate_limit_rpm=30,
    timeout_seconds=30,
    env_vars={"GITHUB_PERSONAL_ACCESS_TOKEN": "${MCP_GITHUB_TOKEN}"},
)


# HuggingFace MCP Server
# Note: The actual MCP server name is "hugging-face" (with hyphen)
HUGGINGFACE_MCP = MCPServerConfig(
    name="hugging-face",  # MCP server uses hyphen
    description="HuggingFace Hub search for models, datasets, papers, and spaces",
    transport=MCPTransport.STDIO,
    command="uvx --from huggingface-mcp-server hf-mcp-server",
    priority=1,
    tools=[
        MCPTool(
            name="model_search",
            description="Search for ML models on HuggingFace Hub",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "task": {"type": "string"},
                    "library": {"type": "string"},
                    "author": {"type": "string"},
                    "sort": {
                        "type": "string",
                        "enum": ["trendingScore", "downloads", "likes"],
                    },
                    "limit": {"type": "integer", "default": 20},
                },
            },
            capabilities=[MCPCapability.SEARCH],
        ),
        MCPTool(
            name="dataset_search",
            description="Search for datasets on HuggingFace Hub",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "author": {"type": "string"},
                    "sort": {
                        "type": "string",
                        "enum": ["trendingScore", "downloads", "likes"],
                    },
                    "limit": {"type": "integer", "default": 20},
                },
            },
            capabilities=[MCPCapability.SEARCH],
        ),
        MCPTool(
            name="paper_search",
            description="Search for ML research papers on HuggingFace",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "results_limit": {"type": "integer", "default": 12},
                    "concise_only": {"type": "boolean", "default": False},
                },
                "required": ["query"],
            },
            capabilities=[MCPCapability.SEARCH],
        ),
        MCPTool(
            name="space_search",
            description="Search for HuggingFace Spaces (demos)",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                    "mcp": {"type": "boolean", "default": False},
                },
                "required": ["query"],
            },
            capabilities=[MCPCapability.SEARCH],
        ),
        MCPTool(
            name="hub_repo_details",
            description="Get details for HuggingFace repos (models, datasets, spaces)",
            input_schema={
                "type": "object",
                "properties": {
                    "repo_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 10,
                    },
                    "repo_type": {
                        "type": "string",
                        "enum": ["model", "dataset", "space"],
                    },
                },
                "required": ["repo_ids"],
            },
            capabilities=[MCPCapability.FETCH],
        ),
    ],
    rate_limit_rpm=60,
    timeout_seconds=30,
)


# AWS Documentation MCP Server (for AWS best practices)
AWS_DOCS_MCP = MCPServerConfig(
    name="aws_docs",
    description="AWS documentation and best practices search",
    transport=MCPTransport.STDIO,
    command="uvx awslabs.aws-documentation-mcp-server",
    enabled=False,  # Enable if needed
    priority=3,
    tools=[
        MCPTool(
            name="search_documentation",
            description="Search AWS documentation",
            input_schema={
                "type": "object",
                "properties": {
                    "search_phrase": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["search_phrase"],
            },
            capabilities=[MCPCapability.SEARCH],
        ),
    ],
    rate_limit_rpm=30,
    timeout_seconds=30,
)


@dataclass
class MCPRegistry:
    """Registry of all available MCP servers."""

    servers: dict[str, MCPServerConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize registry. Does not auto-load built-in STDIO servers."""
        # Built-in STDIO servers require uvx/npx and are disabled by default.
        # Users should configure HTTP MCP servers in ~/.dova.json instead.
        pass

    def get_server(self, name: str) -> MCPServerConfig | None:
        """Get server configuration by name.

        Handles common aliases (e.g., 'hugging-face' and 'huggingface').
        """
        # Try exact match first
        if name in self.servers:
            return self.servers[name]

        # Handle common aliases (normalize hyphenated names)
        aliases = {
            "hugging-face": "huggingface",
            "huggingface": "hugging-face",
        }
        if name in aliases:
            alt_name = aliases[name]
            if alt_name in self.servers:
                return self.servers[alt_name]

        return None

    def get_enabled_servers(self) -> list[MCPServerConfig]:
        """Get all enabled servers."""
        return [s for s in self.servers.values() if s.enabled]

    def get_servers_by_capability(
        self, capability: MCPCapability
    ) -> list[MCPServerConfig]:
        """Get servers that have a specific capability."""
        result = []
        for server in self.servers.values():
            if not server.enabled:
                continue
            for tool in server.tools:
                if capability in tool.capabilities:
                    result.append(server)
                    break
        return result

    def find_tool(self, tool_name: str) -> tuple[MCPServerConfig, MCPTool] | None:
        """Find which server provides a specific tool."""
        for server in self.servers.values():
            if not server.enabled:
                continue
            for tool in server.tools:
                if tool.name == tool_name:
                    return (server, tool)
        return None

    def register_server(self, config: MCPServerConfig) -> None:
        """Register a new MCP server."""
        self.servers[config.name] = config

    def disable_server(self, name: str) -> None:
        """Disable an MCP server."""
        if name in self.servers:
            self.servers[name].enabled = False

    def enable_server(self, name: str) -> None:
        """Enable an MCP server."""
        if name in self.servers:
            self.servers[name].enabled = True


def get_default_registry() -> MCPRegistry:
    """Get the default MCP server registry, loading from ~/.dova.json and managed repos."""
    registry = MCPRegistry()

    # Load user config from ~/.dova.json
    user_servers = load_user_mcp_config()
    for name, config in user_servers.items():
        registry.register_server(config)

    # Load managed MCP repos (e.g., arxiv-mcp-server)
    managed_servers = load_managed_mcp_servers()
    for name, config in managed_servers.items():
        # Don't override user config
        if name not in registry.servers:
            registry.register_server(config)

    return registry


def load_managed_mcp_servers() -> dict[str, MCPServerConfig]:
    """Load MCP server configurations from managed repos (cloned via dova mcp setup)."""
    from pathlib import Path

    servers = {}

    # Check for arxiv-mcp-server
    arxiv_path = Path.home() / ".dova" / "mcp-servers" / "arxiv-mcp-server"
    if arxiv_path.exists():
        storage_path = Path.home() / ".dova" / "arxiv-papers"
        storage_path.mkdir(parents=True, exist_ok=True)

        servers["arxiv"] = MCPServerConfig(
            name="arxiv",
            description="ArXiv paper search and download (blazickjp/arxiv-mcp-server)",
            transport=MCPTransport.STDIO,
            command=f"uv --directory {arxiv_path} run arxiv-mcp-server --storage-path {storage_path}",
            enabled=True,
            priority=1,
            tools=[
                MCPTool(
                    name="search_papers",
                    description="Search ArXiv papers by query",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "max_results": {"type": "integer", "default": 10},
                            "date_from": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                            "categories": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["query"],
                    },
                    capabilities=[MCPCapability.SEARCH],
                ),
                MCPTool(
                    name="download_paper",
                    description="Download a paper by arXiv ID",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "paper_id": {"type": "string", "description": "ArXiv paper ID"},
                        },
                        "required": ["paper_id"],
                    },
                    capabilities=[MCPCapability.FETCH],
                ),
                MCPTool(
                    name="read_paper",
                    description="Read the content of a downloaded paper",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "paper_id": {"type": "string", "description": "ArXiv paper ID"},
                        },
                        "required": ["paper_id"],
                    },
                    capabilities=[MCPCapability.FETCH],
                ),
                MCPTool(
                    name="list_papers",
                    description="List all downloaded papers",
                    input_schema={"type": "object", "properties": {}},
                    capabilities=[MCPCapability.LIST],
                ),
            ],
            rate_limit_rpm=30,
            timeout_seconds=60,
        )

    return servers


def get_dova_config_path() -> str:
    """Get the path to ~/.dova.json."""
    import os

    return os.path.expanduser("~/.dova.json")


def load_user_mcp_config() -> dict[str, MCPServerConfig]:
    """Load MCP server configurations from ~/.dova.json."""
    import json
    import os

    config_path = get_dova_config_path()
    if not os.path.exists(config_path):
        return {}

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

    servers = {}
    mcp_servers = config.get("mcpServers", {})

    for name, server_config in mcp_servers.items():
        transport_type = server_config.get("type", "http")
        if transport_type == "http":
            transport = MCPTransport.HTTP
        elif transport_type == "sse":
            transport = MCPTransport.SSE
        else:
            transport = MCPTransport.STDIO

        servers[name] = MCPServerConfig(
            name=name,
            description=f"User-configured MCP server: {name}",
            transport=transport,
            url=server_config.get("url"),
            headers=server_config.get("headers", {}),
            command=server_config.get("command"),
            enabled=server_config.get("enabled", True),
            priority=server_config.get("priority", 1),
            timeout_seconds=server_config.get("timeout", 30),
        )

    return servers


def save_user_mcp_config(servers: dict[str, dict[str, Any]]) -> None:
    """Save MCP server configurations to ~/.dova.json."""
    import json
    import os

    config_path = get_dova_config_path()

    # Load existing config or create new
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError):
            config = {}
    else:
        config = {}

    # Update mcpServers
    config["mcpServers"] = servers

    # Write config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def add_mcp_server(
    name: str,
    url: str,
    server_type: str = "http",
    headers: dict[str, str] | None = None,
) -> None:
    """Add or update an MCP server in ~/.dova.json."""
    import json
    import os

    config_path = get_dova_config_path()

    # Load existing config
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError):
            config = {}
    else:
        config = {}

    # Ensure mcpServers exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Add/update server
    server_config: dict[str, Any] = {
        "type": server_type,
        "url": url,
    }
    if headers:
        server_config["headers"] = headers

    config["mcpServers"][name] = server_config

    # Write config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def remove_mcp_server(name: str) -> bool:
    """Remove an MCP server from ~/.dova.json."""
    import json
    import os

    config_path = get_dova_config_path()

    if not os.path.exists(config_path):
        return False

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False

    if "mcpServers" not in config or name not in config["mcpServers"]:
        return False

    del config["mcpServers"][name]

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return True


def list_mcp_servers() -> dict[str, dict[str, Any]]:
    """List all MCP servers from ~/.dova.json."""
    import json
    import os

    config_path = get_dova_config_path()

    if not os.path.exists(config_path):
        return {}

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

    return config.get("mcpServers", {})
