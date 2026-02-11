"""
Research Schemas for DOVA API.
"""

from typing import Any

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """Request schema for research queries."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Research query",
        examples=["latest advances in multi-agent LLM systems"],
    )
    sources: list[str] = Field(
        default=["arxiv", "github", "huggingface"],
        description="Sources to search",
    )
    max_results: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum results per source",
    )
    orchestrator: str = Field(
        default="thinking",
        description="Orchestrator type: 'thinking' (deliberation-first) or 'standard' (direct agent)",
    )


class ImageResult(BaseModel):
    """Generated image result."""

    url: str = Field(..., description="URL or path to the generated image")
    prompt: str = Field(..., description="Prompt used to generate the image")
    resolution: str = Field(default="1024x1024", description="Image resolution")
    seed: int = Field(default=0, description="Seed used for generation")


class SearchResult(BaseModel):
    """Individual search result."""

    title: str = Field(..., description="Result title")
    url: str = Field(..., description="Result URL")
    description: str = Field(default="", description="Result description")
    source: str = Field(..., description="Source (arxiv, github, huggingface)")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Relevance score",
    )


class ResearchResponse(BaseModel):
    """Response schema for research queries."""

    query: str = Field(..., description="Original query")
    status: str = Field(..., description="Query status")
    answer: str = Field(default="", description="Direct answer synthesized from research findings")
    summary: str = Field(default="", description="Executive summary of findings")
    papers: list[dict[str, Any]] = Field(
        default_factory=list,
        description="ArXiv papers found",
    )
    repositories: list[dict[str, Any]] = Field(
        default_factory=list,
        description="GitHub repositories found",
    )
    models: list[dict[str, Any]] = Field(
        default_factory=list,
        description="HuggingFace models found",
    )
    datasets: list[dict[str, Any]] = Field(
        default_factory=list,
        description="HuggingFace datasets found",
    )
    web_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Web search results",
    )
    images: list[ImageResult] = Field(
        default_factory=list,
        description="Generated images",
    )
    insights: list[str] = Field(
        default_factory=list,
        description="Key insights from synthesis",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Recommended next steps",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the answer (0.0-1.0)",
    )
    refinement_attempts: int = Field(
        default=0,
        description="Number of query refinements performed",
    )
    reasoning_trace: list[dict[str, Any]] = Field(
        default_factory=list,
        description="ReAct reasoning trace (steps taken during reasoning)",
    )
    debate: dict[str, Any] = Field(
        default_factory=dict,
        description="Bull vs Bear debate analysis (when collaborative mode or evaluative query)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Response metadata (timing, sources, etc.)",
    )


class SearchRequest(BaseModel):
    """Request schema for source-specific search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query",
    )
    max_results: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum results",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific filters",
        examples=[{"language": "python", "min_stars": 100}],
    )


class SearchResponse(BaseModel):
    """Response schema for source-specific search."""

    source: str = Field(..., description="Source searched")
    query: str = Field(..., description="Original query")
    results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Search results",
    )
    total_count: int = Field(default=0, description="Total results found")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Response metadata",
    )


class SynthesisRequest(BaseModel):
    """Request schema for research synthesis."""

    papers: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Papers to synthesize",
    )
    repositories: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Repositories to synthesize",
    )
    models: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Models to synthesize",
    )
    query: str = Field(
        default="",
        description="Original query for context",
    )
    output_format: str = Field(
        default="summary",
        description="Output format (summary, detailed, bullet)",
    )


class SynthesisResponse(BaseModel):
    """Response schema for research synthesis."""

    executive_summary: str = Field(..., description="Executive summary")
    key_findings: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Key findings with evidence",
    )
    knowledge_gaps: list[str] = Field(
        default_factory=list,
        description="Identified knowledge gaps",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Research recommendations",
    )
    cross_references: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Cross-references between artifacts",
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in synthesis",
    )
