"""
DOVA Settings Module.

Pydantic-based configuration with environment variable support.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AWSSettings(BaseSettings):
    """AWS-specific configuration."""

    model_config = SettingsConfigDict(env_prefix="AWS_")

    region: str = Field(default="us-east-1", description="AWS region")
    access_key_id: str | None = Field(default=None, description="AWS access key ID")
    secret_access_key: str | None = Field(default=None, description="AWS secret access key")
    bedrock_model_id: str = Field(
        default="anthropic.claude-sonnet-4-20250514-v1:0",
        description="Default Bedrock model ID",
    )
    agentcore_agent_id: str | None = Field(
        default=None, description="AgentCore deployed agent ID"
    )
    agentcore_agent_alias_id: str | None = Field(
        default=None, description="AgentCore agent alias ID"
    )
    agentcore_memory_enabled: bool = Field(
        default=False, description="Enable AgentCore Memory"
    )


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    default_provider: Literal["bedrock", "anthropic", "openai"] = Field(
        default="bedrock", description="Default LLM provider"
    )
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=200000)


class WebSearchSettings(BaseSettings):
    """Web search provider configuration."""

    model_config = SettingsConfigDict(env_prefix="WEB_SEARCH_")

    provider: Literal["auto", "brave", "perplexity", "tavily", "duckduckgo"] = Field(
        default="auto", description="Web search provider (auto selects best available)"
    )
    brave_api_key: str | None = Field(default=None, description="Brave Search API key")
    perplexity_api_key: str | None = Field(default=None, description="Perplexity API key")
    tavily_api_key: str | None = Field(default=None, description="Tavily API key")
    fallback_enabled: bool = Field(default=True, description="Enable fallback to other providers")

    def __init__(self, **kwargs):
        import os
        # Map common env var names to settings
        env_mappings = [
            ("BRAVE_API_KEY", "WEB_SEARCH_BRAVE_API_KEY"),
            ("PERPLEXITY_API_KEY", "WEB_SEARCH_PERPLEXITY_API_KEY"),
            ("TAVILY_API_KEY", "WEB_SEARCH_TAVILY_API_KEY"),
        ]
        for source, target in env_mappings:
            if os.environ.get(source) and not os.environ.get(target):
                os.environ[target] = os.environ[source]
        super().__init__(**kwargs)


class MCPSettings(BaseSettings):
    """MCP server configuration."""

    model_config = SettingsConfigDict(env_prefix="MCP_")

    arxiv_enabled: bool = Field(default=True, description="Enable ArXiv MCP server")
    github_enabled: bool = Field(default=True, description="Enable GitHub MCP server")
    huggingface_enabled: bool = Field(default=True, description="Enable HuggingFace MCP")
    web_search_enabled: bool = Field(default=True, description="Enable web search via Tavily")
    tavily_api_key: str | None = Field(default=None, description="Tavily API key for web search")
    github_token: str | None = Field(default=None, description="GitHub token for MCP")

    def __init__(self, **kwargs):
        # Check common env var names for Tavily before init
        import os
        for env_name in ["TAVILY_API_KEY", "TAVILY_API_TOKEN", "tavily_api_key", "tavily_api_token"]:
            if os.environ.get(env_name) and not os.environ.get("MCP_TAVILY_API_KEY"):
                os.environ["MCP_TAVILY_API_KEY"] = os.environ[env_name]
                break
        super().__init__(**kwargs)


class AuthSettings(BaseSettings):
    """Authentication configuration."""

    model_config = SettingsConfigDict(env_prefix="AUTH_")

    cognito_user_pool_id: str | None = Field(default=None, description="Cognito user pool ID")
    cognito_client_id: str | None = Field(default=None, description="Cognito client ID")
    kms_key_id: str | None = Field(default=None, description="KMS key ID for credential encryption")
    api_key_expiry_days: int = Field(default=365, description="Default API key expiry in days")


class RedisSettings(BaseSettings):
    """Redis configuration."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535)
    password: str | None = Field(default=None, description="Redis password")
    db: int = Field(default=0, ge=0, le=15)
    ssl: bool = Field(default=False, description="Use SSL for Redis connection")

    @property
    def url(self) -> str:
        """Generate Redis URL."""
        protocol = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"


class APISettings(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, ge=1, le=65535)
    debug: bool = Field(default=False, description="Enable debug mode")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins",
    )
    rate_limit_requests: int = Field(default=100, description="Requests per minute")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


class JobSettings(BaseSettings):
    """Background job configuration."""

    model_config = SettingsConfigDict(env_prefix="JOB_")

    worker_concurrency: int = Field(default=5, ge=1, le=20, description="Worker concurrency")
    arxiv_poll_hours: float = Field(default=1.0, ge=0.1, description="ArXiv poll interval in hours")
    hf_poll_hours: float = Field(default=6.0, ge=0.5, description="HuggingFace poll interval")
    stream_name: str = Field(default="dova:jobs", description="Redis stream name for jobs")
    consumer_group: str = Field(default="dova-workers", description="Redis consumer group name")


class SandboxSettings(BaseSettings):
    """Sandbox execution configuration."""

    model_config = SettingsConfigDict(env_prefix="SANDBOX_")

    enabled: bool = Field(default=False, description="Enable sandbox execution")
    docker_host: str = Field(
        default="unix:///var/run/docker.sock", description="Docker socket path"
    )
    network_enabled: bool = Field(default=False, description="Allow network in sandbox")
    max_concurrent: int = Field(default=5, ge=1, le=20, description="Max concurrent executions")
    max_output_size: int = Field(default=100000, description="Max output size in bytes")
    default_cpu_quota_seconds: int = Field(default=3600, description="Default daily CPU quota")
    default_gpu_quota_seconds: int = Field(default=600, description="Default daily GPU quota")


class ThinkingSettings(BaseSettings):
    """Thinking level configuration for LLM reasoning."""

    model_config = SettingsConfigDict(env_prefix="THINKING_")

    default_level: str = Field(default="medium", description="Default thinking level")
    auto_select_enabled: bool = Field(default=True, description="Auto-select level based on task")
    max_budget_tokens: int = Field(default=65536, description="Maximum thinking budget tokens")


class HeartbeatSettings(BaseSettings):
    """Heartbeat task configuration."""

    model_config = SettingsConfigDict(env_prefix="HEARTBEAT_")

    enabled: bool = Field(default=True, description="Enable heartbeat tasks")
    subscription_monitor_cron: str = Field(
        default="*/15 * * * *", description="Subscription monitor schedule"
    )
    recommendation_refresh_cron: str = Field(
        default="0 */4 * * *", description="Recommendation refresh schedule"
    )
    mcp_health_check_cron: str = Field(
        default="*/5 * * * *", description="MCP health check schedule"
    )
    session_cleanup_cron: str = Field(
        default="0 3 * * *", description="Session cleanup schedule"
    )


class MemoryEnhancedSettings(BaseSettings):
    """Enhanced memory service configuration."""

    model_config = SettingsConfigDict(env_prefix="MEMORY_ENHANCED_")

    semantic_search_enabled: bool = Field(default=True, description="Enable semantic search")
    mmr_lambda: float = Field(
        default=0.5, ge=0.0, le=1.0, description="MMR diversity parameter"
    )
    embedding_cache_ttl: int = Field(default=3600, description="Embedding cache TTL in seconds")


class DiscoverySettings(BaseSettings):
    """Auto-discovery service configuration."""

    model_config = SettingsConfigDict(env_prefix="DISCOVERY_")

    auto_discover_on_startup: bool = Field(default=True, description="Discover on startup")
    cache_ttl_seconds: int = Field(default=3600, description="Discovery cache TTL")


class EvaluationSettings(BaseSettings):
    """Self-evaluation service configuration."""

    model_config = SettingsConfigDict(env_prefix="EVAL_")

    auto_evaluate_responses: bool = Field(default=False, description="Auto-evaluate responses")
    min_confidence_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )


class SessionSettings(BaseSettings):
    """Session management configuration."""

    model_config = SettingsConfigDict(env_prefix="SESSION_")

    stale_after_seconds: int = Field(default=1800, description="Session stale timeout (30 min)")
    expire_after_seconds: int = Field(default=86400, description="Session expiry timeout (24h)")


class AgentCoreSettings(BaseSettings):
    """AgentCore-specific settings for AWS Bedrock integration."""

    model_config = SettingsConfigDict(env_prefix="AGENTCORE_")

    stack_name: str = Field(default="", description="CloudFormation stack name")
    memory_id: str = Field(default="", description="AgentCore Memory ID")
    gateway_url: str = Field(default="", description="AgentCore Gateway URL")
    runtime_mode: str = Field(
        default="fastapi",
        description="Runtime mode: 'fastapi' or 'agentcore'",
    )


class MemorySettings(BaseSettings):
    """Memory strategy configuration for AgentCore Memory."""

    model_config = SettingsConfigDict(env_prefix="MEMORY_")

    summary_enabled: bool = Field(default=False, description="Enable summary memory retrieval")
    user_preference_enabled: bool = Field(
        default=False, description="Enable user preference memory retrieval"
    )
    semantic_enabled: bool = Field(default=False, description="Enable semantic memory retrieval")

    # Retrieval configs
    summary_top_k: int = Field(default=5, ge=1, le=50, description="Number of summaries to retrieve")
    preference_top_k: int = Field(
        default=5, ge=1, le=50, description="Number of preferences to retrieve"
    )
    semantic_top_k: int = Field(
        default=10, ge=1, le=100, description="Number of semantic memories to retrieve"
    )
    semantic_relevance: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum relevance score for semantic retrieval"
    )


class Settings(BaseSettings):
    """Main DOVA settings aggregating all sub-configurations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application metadata
    app_name: str = Field(default="DOVA", description="Application name")
    app_version: str = Field(default="1.1.0", description="Application version")
    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Deployment environment"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )

    # Sub-configurations
    aws: AWSSettings = Field(default_factory=AWSSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    api: APISettings = Field(default_factory=APISettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    jobs: JobSettings = Field(default_factory=JobSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    thinking: ThinkingSettings = Field(default_factory=ThinkingSettings)
    heartbeat: HeartbeatSettings = Field(default_factory=HeartbeatSettings)
    memory_enhanced: MemoryEnhancedSettings = Field(default_factory=MemoryEnhancedSettings)
    discovery: DiscoverySettings = Field(default_factory=DiscoverySettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    session: SessionSettings = Field(default_factory=SessionSettings)
    agentcore: AgentCoreSettings = Field(default_factory=AgentCoreSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
