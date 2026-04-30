"""
Research Tools for DOVA.

Provides Strands SDK-compatible tools for research operations.
These tools can be used directly with Strands Agents.
"""

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def search_arxiv_tool(
    query: str,
    max_results: int = 10,
    category: str | None = None,
    author: str | None = None,
) -> dict[str, Any]:
    """
    Search ArXiv for academic papers.

    Args:
        query: Search query string
        max_results: Maximum number of results (default 10, max 100)
        category: Optional ArXiv category filter (e.g., 'cs.AI', 'cs.LG')
        author: Optional author name filter

    Returns:
        Dictionary with search results including:
        - papers: List of paper objects with title, authors, abstract, url
        - total: Total number of results found
        - query_used: The actual query sent to ArXiv
    """
    # This is a tool definition for Strands SDK
    # The actual implementation calls the MCP server

    # Build search query
    query_parts = [query]
    if category:
        query_parts.append(f"cat:{category}")
    if author:
        query_parts.append(f"au:{author}")
    search_query = " AND ".join(query_parts)

    logger.info("arxiv_search", query=search_query, max_results=max_results)

    # Return tool schema for Strands
    return {
        "tool_name": "search_papers",  # blazickjp/arxiv-mcp-server tool name
        "mcp_server": "arxiv",
        "params": {
            "query": search_query,
            "max_results": min(max_results, 50),  # Server max is 50
        },
    }


def search_github_tool(
    query: str,
    search_type: str = "repositories",
    language: str | None = None,
    min_stars: int | None = None,
    per_page: int = 30,
) -> dict[str, Any]:
    """
    Search GitHub for repositories or code.

    Args:
        query: Search query string
        search_type: Type of search - 'repositories' or 'code'
        language: Optional programming language filter
        min_stars: Optional minimum star count for repositories
        per_page: Number of results per page (default 30, max 100)

    Returns:
        Dictionary with search results including:
        - items: List of repository/code objects
        - total_count: Total number of matching items
        - query_used: The actual query sent to GitHub
    """
    # Build search query with filters
    query_parts = [query]
    if language:
        query_parts.append(f"language:{language}")
    if min_stars is not None:
        query_parts.append(f"stars:>={min_stars}")

    search_query = " ".join(query_parts)

    logger.info(
        "github_search",
        query=search_query,
        search_type=search_type,
        per_page=per_page,
    )

    return {
        "tool_name": f"search_{search_type}",
        "mcp_server": "github",
        "params": {
            "query": search_query,
            "per_page": min(per_page, 100),
        },
    }


def search_huggingface_tool(
    query: str,
    search_type: str = "models",
    task: str | None = None,
    library: str | None = None,
    sort_by: str = "downloads",
    limit: int = 20,
) -> dict[str, Any]:
    """
    Search HuggingFace Hub for models, datasets, or papers.

    Args:
        query: Search query string
        search_type: Type of search - 'models', 'datasets', or 'papers'
        task: Optional task filter (e.g., 'text-generation', 'image-classification')
        library: Optional library filter (e.g., 'transformers', 'diffusers')
        sort_by: Sort order - 'downloads', 'likes', or 'trending'
        limit: Maximum number of results (default 20, max 100)

    Returns:
        Dictionary with search results including:
        - items: List of model/dataset/paper objects
        - total: Total number of results
    """
    logger.info(
        "huggingface_search",
        query=query,
        search_type=search_type,
        task=task,
        library=library,
    )

    params: dict[str, Any] = {"query": query, "limit": min(limit, 100)}

    if search_type in ("models", "datasets"):
        tool_name = "hub_repo_search"
        params["sort"] = sort_by
    elif search_type == "papers":
        tool_name = "paper_search"
        params["results_limit"] = limit
    else:
        tool_name = "hub_repo_search"

    return {
        "tool_name": tool_name,
        "mcp_server": "hugging-face",  # MCP server name uses hyphen
        "params": params,
    }


def search_bio_tool(
    query: str,
    domain: str = "auto",
    max_results: int = 10,
) -> dict[str, Any]:
    """
    Search biomedical / pharma sources (PubMed, ClinicalTrials.gov, PubChem).

    Routes the query to the most relevant hosted MCP server based on the
    `domain` hint (or keyword auto-detection when `domain="auto"`).

    Args:
        query: Search query string.
        domain: One of 'literature' (PubMed), 'trials' (ClinicalTrials.gov),
                'compounds' (PubChem), or 'auto' for keyword-based routing.
        max_results: Maximum number of results (default 10).

    Returns:
        Dispatch record with tool_name, mcp_server, and params — the
        ThinkingOrchestrator performs the actual routing.
    """
    domain_map = {
        "literature": ("pubmed-bio", "pubmed_search_articles"),
        "pubmed": ("pubmed-bio", "pubmed_search_articles"),
        "trials": ("clinicaltrials-bio", "clinicaltrials_search_studies"),
        "clinical": ("clinicaltrials-bio", "clinicaltrials_search_studies"),
        "compounds": ("pubchem-bio", "pubchem_search_compounds"),
        "chemicals": ("pubchem-bio", "pubchem_search_compounds"),
        "drugs": ("pubchem-bio", "pubchem_search_compounds"),
    }

    if domain == "auto":
        mcp_server, tool_name = "pubmed-bio", "pubmed_search_articles"
    else:
        mcp_server, tool_name = domain_map.get(
            domain.lower(), ("pubmed-bio", "pubmed_search_articles")
        )

    logger.info("bio_search", query=query, domain=domain, mcp_server=mcp_server)

    capped = min(max_results, 50)
    if mcp_server == "pubmed-bio":
        params: dict[str, Any] = {"query": query, "maxResults": capped}
    elif mcp_server == "clinicaltrials-bio":
        params = {"query": query}
    else:  # pubchem-bio — schema: searchType + identifierType + identifiers[]
        params = {
            "searchType": "identifier",
            "identifierType": "name",
            "identifiers": [query],
        }

    return {
        "tool_name": tool_name,
        "mcp_server": mcp_server,
        "params": params,
    }


def web_search_tool(
    query: str,
    search_depth: str = "basic",
    max_results: int = 10,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> dict[str, Any]:
    """
    Search the web using Tavily.

    Args:
        query: Search query string
        search_depth: Search depth - 'basic' or 'advanced'
        max_results: Maximum number of results (default 10)
        include_domains: Optional list of domains to include
        exclude_domains: Optional list of domains to exclude

    Returns:
        Dictionary with search results including:
        - results: List of web page objects with title, url, content
        - answer: AI-generated answer if available
    """
    logger.info(
        "web_search",
        query=query,
        search_depth=search_depth,
        max_results=max_results,
    )

    params: dict[str, Any] = {
        "query": query,
        "search_depth": search_depth,
        "max_results": max_results,
    }

    if include_domains:
        params["include_domains"] = include_domains
    if exclude_domains:
        params["exclude_domains"] = exclude_domains

    return {
        "tool_name": "tavily_search",
        "builtin": True,  # Tavily is a built-in Strands tool
        "params": params,
    }


def get_github_file_tool(
    owner: str,
    repo: str,
    path: str,
    ref: str | None = None,
) -> dict[str, Any]:
    """
    Get contents of a file from a GitHub repository.

    Args:
        owner: Repository owner (user or organization)
        repo: Repository name
        path: Path to file within repository
        ref: Optional git ref (branch, tag, or commit SHA)

    Returns:
        Dictionary with file contents and metadata
    """
    logger.info(
        "github_get_file",
        owner=owner,
        repo=repo,
        path=path,
        ref=ref,
    )

    params: dict[str, Any] = {
        "owner": owner,
        "repo": repo,
        "path": path,
    }
    if ref:
        params["ref"] = ref

    return {
        "tool_name": "get_file_contents",
        "mcp_server": "github",
        "params": params,
    }


def get_huggingface_model_tool(
    model_id: str,
) -> dict[str, Any]:
    """
    Get details for a HuggingFace model.

    Args:
        model_id: Model ID (e.g., 'meta-llama/Llama-2-7b-hf')

    Returns:
        Dictionary with model details including:
        - config: Model configuration
        - downloads: Download count
        - tags: Model tags
        - pipeline_tag: Task the model is designed for
    """
    logger.info("huggingface_get_model", model_id=model_id)

    return {
        "tool_name": "hub_repo_details",
        "mcp_server": "hugging-face",  # MCP server name uses hyphen
        "params": {
            "repo_ids": [model_id],
            "repo_type": "model",
        },
    }


# Tool definitions for Strands SDK registration
RESEARCH_TOOLS = [
    {
        "name": "search_arxiv",
        "description": "Search ArXiv for academic papers on a topic",
        "function": search_arxiv_tool,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for papers",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 10,
                },
                "category": {
                    "type": "string",
                    "description": "ArXiv category (e.g., cs.AI, cs.LG)",
                },
                "author": {
                    "type": "string",
                    "description": "Filter by author name",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_github",
        "description": "Search GitHub for repositories or code",
        "function": search_github_tool,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["repositories", "code"],
                    "default": "repositories",
                },
                "language": {
                    "type": "string",
                    "description": "Programming language filter",
                },
                "min_stars": {
                    "type": "integer",
                    "description": "Minimum star count",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_huggingface",
        "description": "Search HuggingFace Hub for models, datasets, or papers",
        "function": search_huggingface_tool,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["models", "datasets", "papers"],
                    "default": "models",
                },
                "task": {
                    "type": "string",
                    "description": "Task filter (e.g., text-generation)",
                },
                "library": {
                    "type": "string",
                    "description": "Library filter (e.g., transformers)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_bio",
        "description": "Search biomedical / pharma data (PubMed literature, ClinicalTrials.gov trials, PubChem compounds)",
        "function": search_bio_tool,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "domain": {
                    "type": "string",
                    "enum": ["auto", "literature", "trials", "compounds"],
                    "default": "auto",
                    "description": "Which bio subdomain to target",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web for information",
        "function": web_search_tool,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
]
