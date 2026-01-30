"""DOVA Custom Tools Module."""

from dova.tools.mcp_registry import MCPClient, MCPManager
from dova.tools.research_tools import (
    search_arxiv_tool,
    search_github_tool,
    search_huggingface_tool,
    web_search_tool,
)
from dova.tools.synthesis_tools import (
    synthesize_research_tool,
    cross_reference_tool,
    generate_summary_tool,
)

__all__ = [
    "MCPClient",
    "MCPManager",
    "search_arxiv_tool",
    "search_github_tool",
    "search_huggingface_tool",
    "web_search_tool",
    "synthesize_research_tool",
    "cross_reference_tool",
    "generate_summary_tool",
]
