"""
DOVA FastAPI Application.

Main entry point for the DOVA REST API.
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dova.api.middleware.logging import LoggingMiddleware
from dova.api.middleware.rate_limit import RateLimitMiddleware
from dova.api.routes import (
    credentials,
    health,
    memory,
    profile,
    research,
    sources,
    subscriptions,
    validation,
    webhooks,
)
from dova.config.settings import Settings, get_settings
from dova.services.api_key_service import APIKeyService
from dova.services.jwt_verifier import CognitoJWTVerifier, JWTVerifierConfig
from dova.services.memory import AgentCoreMemoryService
from dova.services.sources import SourceRegistry
from dova.utils.logging import configure_logging

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan events."""
    settings = get_settings()

    # Startup
    configure_logging(
        log_level=settings.log_level,
        json_logs=settings.is_production,
        service_name=settings.app_name,
    )
    logger.info(
        "dova_startup",
        environment=settings.environment,
        version=settings.app_version,
    )

    # Initialize services
    app.state.settings = settings

    # Initialize memory service if enabled
    if settings.aws.agentcore_memory_enabled and settings.aws.agentcore_agent_id:
        app.state.memory_service = AgentCoreMemoryService(
            agent_id=settings.aws.agentcore_agent_id,
            agent_alias_id=settings.aws.agentcore_agent_alias_id or "",
            region=settings.aws.region,
        )
    else:
        app.state.memory_service = None

    # Initialize source registry
    app.state.source_registry = SourceRegistry(
        memory_service=app.state.memory_service
    )

    # Initialize LLM router and MCP client
    from dova.config.providers import create_llm_router_from_settings
    from dova.tools.mcp_registry import MCPManager

    app.state.llm_router = create_llm_router_from_settings()
    app.state.mcp_manager = MCPManager()
    app.state.mcp_client = app.state.mcp_manager.get_client()
    logger.info("llm_router_initialized", default_provider=settings.llm.default_provider)

    # Setup MCP server repos (clone/update arxiv-mcp-server etc.)
    from dova.services.mcp_repo_manager import setup_mcp_repos

    try:
        mcp_repo_results = await setup_mcp_repos()
        logger.info("mcp_repos_setup", results=mcp_repo_results)
    except Exception as e:
        logger.warning("mcp_repos_setup_failed", error=str(e))

    # Initialize research agent
    from dova.agents.research import ResearchAgent

    app.state.research_agent = ResearchAgent(
        llm_router=app.state.llm_router,
        mcp_client=app.state.mcp_client,
        memory_service=app.state.memory_service,
        source_registry=app.state.source_registry,
        tavily_api_key=settings.mcp.tavily_api_key,
    )
    logger.info(
        "research_agent_initialized",
        web_search_enabled=bool(settings.mcp.tavily_api_key),
    )

    # Initialize JWT verifier if Cognito is configured
    if settings.auth.cognito_user_pool_id and settings.auth.cognito_client_id:
        app.state.jwt_verifier = CognitoJWTVerifier(
            JWTVerifierConfig(
                region=settings.aws.region,
                user_pool_id=settings.auth.cognito_user_pool_id,
                client_id=settings.auth.cognito_client_id,
            )
        )
        logger.info("jwt_verifier_initialized", user_pool_id=settings.auth.cognito_user_pool_id)
    else:
        app.state.jwt_verifier = None

    # Initialize API key service if KMS and memory are configured
    if settings.auth.kms_key_id and app.state.memory_service:
        app.state.api_key_service = APIKeyService(
            memory_service=app.state.memory_service,
            kms_key_id=settings.auth.kms_key_id,
            region=settings.aws.region,
        )
        logger.info("api_key_service_initialized", kms_key_id=settings.auth.kms_key_id)
    else:
        app.state.api_key_service = None

    # Initialize recommendation services
    from dova.services.recommendation.delivery import DeliveryManager
    from dova.services.recommendation.subscriptions import SubscriptionManager

    app.state.subscription_manager = SubscriptionManager(
        memory_service=app.state.memory_service
    )
    app.state.delivery_manager = DeliveryManager(
        memory_service=app.state.memory_service
    )
    logger.info("recommendation_services_initialized")

    # Initialize job queue and scheduler
    from redis.asyncio import Redis

    from dova.jobs.scheduler import DOVAScheduler
    from dova.jobs.streams import JobQueue

    redis_client = Redis.from_url(settings.redis.url)
    app.state.redis = redis_client
    app.state.job_queue = JobQueue(
        redis_client,
        stream_name=settings.jobs.stream_name,
        group_name=settings.jobs.consumer_group,
    )

    scheduler = DOVAScheduler(
        app.state.job_queue,
        arxiv_poll_hours=settings.jobs.arxiv_poll_hours,
        hf_poll_hours=settings.jobs.hf_poll_hours,
    )
    app.state.scheduler = scheduler

    # Start scheduler in background
    try:
        await scheduler.start()
        logger.info("job_scheduler_started")
    except Exception as e:
        logger.warning("job_scheduler_start_error", error=str(e))

    # Initialize heartbeat processor for proactive tasks (weekly MCP updates, etc.)
    from dova.jobs.heartbeat import HeartbeatProcessor

    try:
        app.state.heartbeat = HeartbeatProcessor(
            job_queue=app.state.job_queue,
            auto_register_defaults=True,
        )
        await app.state.heartbeat.start()
        logger.info(
            "heartbeat_started",
            tasks=[t.name for t in app.state.heartbeat.list_tasks()],
        )
    except Exception as e:
        logger.warning("heartbeat_start_error", error=str(e))
        app.state.heartbeat = None

    # Initialize sandbox executor if enabled
    if settings.sandbox.enabled:
        from dova.services.sandbox.executor import DockerExecutor
        from dova.services.sandbox.quota import QuotaManager
        from dova.services.sandbox.scheduler import SandboxScheduler

        app.state.sandbox_scheduler = SandboxScheduler()
        app.state.sandbox_executor = DockerExecutor(
            docker_host=settings.sandbox.docker_host,
            network_enabled=settings.sandbox.network_enabled,
            max_output_size=settings.sandbox.max_output_size,
        )
        app.state.quota_manager = QuotaManager(
            memory_service=app.state.memory_service,
            default_cpu_seconds=settings.sandbox.default_cpu_quota_seconds,
            default_gpu_seconds=settings.sandbox.default_gpu_quota_seconds,
        )
        logger.info("sandbox_services_initialized")
    else:
        app.state.sandbox_scheduler = None
        app.state.sandbox_executor = None
        app.state.quota_manager = None

    yield

    # Shutdown
    if hasattr(app.state, "heartbeat") and app.state.heartbeat:
        await app.state.heartbeat.stop()
    if hasattr(app.state, "scheduler") and app.state.scheduler:
        await app.state.scheduler.stop()
    if hasattr(app.state, "redis") and app.state.redis:
        await app.state.redis.close()
    logger.info("dova_shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        settings: Optional settings override

    Returns:
        Configured FastAPI application
    """
    settings = settings or get_settings()

    app = FastAPI(
        title="DOVA API",
        description="Deep Orchestrated Versatile Agent Platform - Research Automation API",
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        openapi_url="/openapi.json" if settings.is_development else None,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.api.rate_limit_requests,
        window_seconds=settings.api.rate_limit_window,
    )

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(research.router, prefix="/api/v1", tags=["Research"])
    app.include_router(profile.router, prefix="/api/v1", tags=["Profile"])
    app.include_router(validation.router, prefix="/api/v1", tags=["Validation"])
    app.include_router(memory.router, prefix="/api/v1", tags=["Memory"])
    app.include_router(sources.router, prefix="/api/v1", tags=["Sources"])
    app.include_router(credentials.router, prefix="/api/v1", tags=["Credentials"])
    app.include_router(subscriptions.router, prefix="/api/v1", tags=["Subscriptions"])
    app.include_router(webhooks.router, prefix="/api/v1", tags=["Webhooks"])

    return app


# Default application instance
app = create_app()


def run() -> None:
    """Run the API server using uvicorn."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "dova.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
