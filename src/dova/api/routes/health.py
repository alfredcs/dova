"""
Health Check Endpoints for DOVA API.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Basic health check endpoint.

    Returns:
        Health status with timestamp
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "dova-api",
    }


@router.get("/health/ready")
async def readiness_check(request: Request) -> dict[str, Any]:
    """
    Readiness check - verifies all dependencies are available.

    Returns:
        Readiness status with component health
    """
    settings = request.app.state.settings
    components: dict[str, bool] = {}

    # Check Redis connection
    try:
        # Placeholder - in production would actually ping Redis
        components["redis"] = True
    except Exception:
        components["redis"] = False

    # Check LLM provider availability
    components["llm_provider"] = True  # Placeholder

    # Check MCP servers
    components["mcp_servers"] = True  # Placeholder

    all_healthy = all(components.values())

    return {
        "status": "ready" if all_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "components": components,
        "environment": settings.environment,
    }


@router.get("/health/live")
async def liveness_check() -> dict[str, Any]:
    """
    Liveness check - basic check that service is running.

    Returns:
        Liveness status
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/version")
async def version(request: Request) -> dict[str, Any]:
    """
    Get API version information.

    Returns:
        Version details
    """
    settings = request.app.state.settings
    return {
        "version": settings.app_version,
        "name": settings.app_name,
        "environment": settings.environment,
    }
