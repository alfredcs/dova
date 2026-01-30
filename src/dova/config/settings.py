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


class MCPSettings(BaseSettings):
    """MCP server configuration."""

    model_config = SettingsConfigDict(env_prefix="MCP_")

    arxiv_enabled: bool = Field(default=True, description="Enable ArXiv MCP server")
    github_enabled: bool = Field(default=True, description="Enable GitHub MCP server")
    huggingface_enabled: bool = Field(default=True, description="Enable HuggingFace MCP")
    tavily_api_key: str | None = Field(default=None, description="Tavily API key for web search")
    github_token: str | None = Field(default=None, description="GitHub token for MCP")


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


class Settings(BaseSettings):
    """Main DOVA settings aggregating all sub-configurations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application metadata
    app_name: str = Field(default="DOVA", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
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
    redis: RedisSettings = Field(default_factory=RedisSettings)
    api: APISettings = Field(default_factory=APISettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    jobs: JobSettings = Field(default_factory=JobSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)

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
