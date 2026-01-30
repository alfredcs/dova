"""
Synthesis Tools for DOVA.

Provides tools for synthesizing and summarizing research findings.
"""

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def synthesize_research_tool(
    papers: list[dict[str, Any]] | None = None,
    repositories: list[dict[str, Any]] | None = None,
    models: list[dict[str, Any]] | None = None,
    query: str = "",
    output_format: str = "summary",
) -> dict[str, Any]:
    """
    Synthesize research findings from multiple sources.

    Args:
        papers: List of paper objects from ArXiv/HuggingFace
        repositories: List of repository objects from GitHub
        models: List of model objects from HuggingFace
        query: Original research query for context
        output_format: Output format - 'summary', 'detailed', or 'bullet'

    Returns:
        Dictionary with synthesis results including:
        - synthesis: Main synthesized content
        - key_themes: Identified themes across sources
        - connections: Cross-source connections found
    """
    logger.info(
        "synthesize_research",
        paper_count=len(papers or []),
        repo_count=len(repositories or []),
        model_count=len(models or []),
        query=query,
    )

    # Prepare data for synthesis
    synthesis_data = {
        "papers": _format_papers_for_synthesis(papers or []),
        "repositories": _format_repos_for_synthesis(repositories or []),
        "models": _format_models_for_synthesis(models or []),
        "query": query,
        "output_format": output_format,
    }

    return {
        "tool_name": "synthesize",
        "internal": True,
        "data": synthesis_data,
        "prompt_template": """Synthesize these research findings:

Query: {query}

Papers:
{papers}

Repositories:
{repositories}

Models:
{models}

Provide a {output_format} synthesis including:
1. Key themes and findings
2. Connections between papers, code, and models
3. Research gaps and opportunities
4. Recommendations for further exploration""",
    }


def cross_reference_tool(
    papers: list[dict[str, Any]] | None = None,
    repositories: list[dict[str, Any]] | None = None,
    models: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Find cross-references between papers, code, and models.

    Args:
        papers: List of paper objects
        repositories: List of repository objects
        models: List of model objects

    Returns:
        Dictionary with cross-reference mappings:
        - paper_to_code: Papers with implementations
        - code_to_model: Repos that trained models
        - model_to_paper: Models citing papers
    """
    logger.info(
        "cross_reference",
        paper_count=len(papers or []),
        repo_count=len(repositories or []),
        model_count=len(models or []),
    )

    return {
        "tool_name": "cross_reference",
        "internal": True,
        "data": {
            "papers": papers or [],
            "repositories": repositories or [],
            "models": models or [],
        },
        "prompt_template": """Analyze these research artifacts and find cross-references:

Papers: {papers}
Repositories: {repositories}
Models: {models}

Identify:
1. Papers that have code implementations in the repositories
2. Repositories that produced or fine-tuned models
3. Models that reference specific papers
4. Shared datasets across papers and models

Output as JSON with keys: paper_to_code, code_to_model, model_to_paper, shared_resources""",
    }


def generate_summary_tool(
    content: str,
    summary_type: str = "executive",
    max_length: int = 500,
    audience: str = "technical",
) -> dict[str, Any]:
    """
    Generate a summary of research content.

    Args:
        content: Content to summarize
        summary_type: Type of summary - 'executive', 'technical', 'key_points'
        max_length: Maximum length in words
        audience: Target audience - 'technical', 'business', 'general'

    Returns:
        Dictionary with summary parameters for LLM processing
    """
    logger.info(
        "generate_summary",
        summary_type=summary_type,
        max_length=max_length,
        audience=audience,
        content_length=len(content),
    )

    style_guides = {
        "technical": "Use precise technical language and include methodology details.",
        "business": "Focus on business implications, ROI, and practical applications.",
        "general": "Use accessible language and avoid jargon.",
    }

    type_guides = {
        "executive": "Provide a high-level overview with key takeaways.",
        "technical": "Include technical details, methods, and results.",
        "key_points": "Format as bullet points with the most important findings.",
    }

    return {
        "tool_name": "summarize",
        "internal": True,
        "data": {
            "content": content,
            "max_length": max_length,
        },
        "prompt_template": f"""Summarize the following content:

{{content}}

Requirements:
- Summary type: {summary_type} - {type_guides.get(summary_type, '')}
- Target audience: {audience} - {style_guides.get(audience, '')}
- Maximum length: {max_length} words

Provide a clear, well-structured summary.""",
    }


def extract_key_findings_tool(
    papers: list[dict[str, Any]],
    focus_areas: list[str] | None = None,
) -> dict[str, Any]:
    """
    Extract key findings from a set of papers.

    Args:
        papers: List of paper objects with abstracts
        focus_areas: Optional list of areas to focus on

    Returns:
        Dictionary with extraction parameters
    """
    logger.info(
        "extract_key_findings",
        paper_count=len(papers),
        focus_areas=focus_areas,
    )

    return {
        "tool_name": "extract_findings",
        "internal": True,
        "data": {
            "papers": papers,
            "focus_areas": focus_areas or [],
        },
        "prompt_template": """Extract key findings from these papers:

{papers}

Focus areas: {focus_areas}

For each paper, extract:
1. Main contribution/finding
2. Methodology summary
3. Key results/metrics
4. Limitations noted
5. Future work suggested

Then synthesize across papers:
- Common themes
- Contradictions or debates
- Emerging trends
- Research gaps""",
    }


def compare_approaches_tool(
    approaches: list[dict[str, str]],
    comparison_criteria: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compare different research approaches or methods.

    Args:
        approaches: List of approach descriptions with name and description
        comparison_criteria: Criteria for comparison (e.g., 'accuracy', 'efficiency')

    Returns:
        Dictionary with comparison parameters
    """
    default_criteria = [
        "accuracy/performance",
        "computational efficiency",
        "ease of implementation",
        "scalability",
        "interpretability",
    ]

    criteria = comparison_criteria or default_criteria

    logger.info(
        "compare_approaches",
        approach_count=len(approaches),
        criteria=criteria,
    )

    return {
        "tool_name": "compare_approaches",
        "internal": True,
        "data": {
            "approaches": approaches,
            "criteria": criteria,
        },
        "prompt_template": """Compare these research approaches:

{approaches}

Comparison criteria:
{criteria}

Provide:
1. Comparison table with each approach vs each criterion
2. Pros and cons for each approach
3. Best use cases for each
4. Overall recommendation based on common scenarios""",
    }


def _format_papers_for_synthesis(papers: list[dict[str, Any]]) -> str:
    """Format papers for synthesis prompt."""
    if not papers:
        return "No papers provided."

    formatted = []
    for i, paper in enumerate(papers[:10], 1):
        title = paper.get("title", "Untitled")
        abstract = paper.get("abstract", paper.get("summary", ""))[:300]
        authors = paper.get("authors", [])
        author_str = ", ".join(authors[:3]) if authors else "Unknown"

        formatted.append(f"{i}. {title}\n   Authors: {author_str}\n   Abstract: {abstract}...")

    return "\n\n".join(formatted)


def _format_repos_for_synthesis(repos: list[dict[str, Any]]) -> str:
    """Format repositories for synthesis prompt."""
    if not repos:
        return "No repositories provided."

    formatted = []
    for i, repo in enumerate(repos[:10], 1):
        name = repo.get("full_name", repo.get("name", "Unknown"))
        description = repo.get("description", "")[:200]
        stars = repo.get("stargazers_count", repo.get("stars", 0))
        language = repo.get("language", "Unknown")

        formatted.append(
            f"{i}. {name} ({stars} stars, {language})\n   {description}"
        )

    return "\n\n".join(formatted)


def _format_models_for_synthesis(models: list[dict[str, Any]]) -> str:
    """Format models for synthesis prompt."""
    if not models:
        return "No models provided."

    formatted = []
    for i, model in enumerate(models[:10], 1):
        model_id = model.get("id", model.get("modelId", "Unknown"))
        downloads = model.get("downloads", 0)
        pipeline_tag = model.get("pipeline_tag", "")
        tags = model.get("tags", [])[:5]

        formatted.append(
            f"{i}. {model_id} ({downloads} downloads)\n   Task: {pipeline_tag}\n   Tags: {', '.join(tags)}"
        )

    return "\n\n".join(formatted)


# Tool definitions for Strands SDK registration
SYNTHESIS_TOOLS = [
    {
        "name": "synthesize_research",
        "description": "Synthesize research findings from papers, code, and models",
        "function": synthesize_research_tool,
        "parameters": {
            "type": "object",
            "properties": {
                "papers": {
                    "type": "array",
                    "description": "List of paper objects",
                },
                "repositories": {
                    "type": "array",
                    "description": "List of repository objects",
                },
                "models": {
                    "type": "array",
                    "description": "List of model objects",
                },
                "query": {
                    "type": "string",
                    "description": "Original research query",
                },
                "output_format": {
                    "type": "string",
                    "enum": ["summary", "detailed", "bullet"],
                    "default": "summary",
                },
            },
        },
    },
    {
        "name": "cross_reference",
        "description": "Find cross-references between papers, code, and models",
        "function": cross_reference_tool,
        "parameters": {
            "type": "object",
            "properties": {
                "papers": {"type": "array"},
                "repositories": {"type": "array"},
                "models": {"type": "array"},
            },
        },
    },
    {
        "name": "generate_summary",
        "description": "Generate a summary of research content",
        "function": generate_summary_tool,
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Content to summarize",
                },
                "summary_type": {
                    "type": "string",
                    "enum": ["executive", "technical", "key_points"],
                    "default": "executive",
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum length in words",
                    "default": 500,
                },
                "audience": {
                    "type": "string",
                    "enum": ["technical", "business", "general"],
                    "default": "technical",
                },
            },
            "required": ["content"],
        },
    },
]
