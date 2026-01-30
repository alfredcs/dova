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
            name="search_arxiv",
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
HUGGINGFACE_MCP = MCPServerConfig(
    name="huggingface",
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
        """Initialize with default servers."""
        if not self.servers:
            self.servers = {
                "arxiv": ARXIV_MCP,
                "github": GITHUB_MCP,
                "huggingface": HUGGINGFACE_MCP,
                "aws_docs": AWS_DOCS_MCP,
            }

    def get_server(self, name: str) -> MCPServerConfig | None:
        """Get server configuration by name."""
        return self.servers.get(name)

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
    """Get the default MCP server registry."""
    return MCPRegistry()
