"""
Authentication Middleware for DOVA API.

Supports JWT (Cognito) and API key authentication.
"""

from dataclasses import dataclass
from typing import Any

import structlog
from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from dova.config.settings import get_settings

logger = structlog.get_logger(__name__)

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass
class User:
    """Authenticated user."""

    id: str
    email: str
    roles: list[str]
    metadata: dict[str, Any]


async def get_current_user(
    request: Request,
    bearer_token: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    api_key: str | None = Security(api_key_header),
) -> User:
    """
    Get the current authenticated user.

    Supports both JWT (Bearer token) and API key authentication.

    Args:
        request: FastAPI request (for accessing app.state services)
        bearer_token: Optional Bearer token from Authorization header
        api_key: Optional API key from X-API-Key header

    Returns:
        Authenticated User object

    Raises:
        HTTPException: If authentication fails in production
    """
    settings = get_settings()

    # Try JWT (Cognito) first
    if bearer_token:
        jwt_verifier = getattr(request.app.state, "jwt_verifier", None)
        if jwt_verifier:
            claims = await jwt_verifier.verify(bearer_token.credentials)
            if claims:
                user = User(
                    id=claims["sub"],
                    email=claims.get("email", ""),
                    roles=claims.get("cognito:groups", []),
                    metadata={"token_type": "jwt"},
                )
                logger.debug("user_authenticated", user_id=user.id, method="jwt")
                return user

    # Try API key
    if api_key:
        api_key_service = getattr(request.app.state, "api_key_service", None)
        if api_key_service:
            record = await api_key_service.verify_dova_api_key(api_key)
            if record:
                user = User(
                    id=record.user_id,
                    email="",
                    roles=record.roles,
                    metadata={"token_type": "api_key", "key_id": record.key_id},
                )
                logger.debug("user_authenticated", user_id=user.id, method="api_key")
                return user
        # Development fallback for API keys when service not initialized
        elif settings.is_development and api_key.startswith("dova_"):
            user = User(
                id=f"dev_{api_key[:16]}",
                email="",
                roles=["api_user"],
                metadata={"token_type": "api_key", "dev_mode": True},
            )
            logger.debug("user_authenticated", user_id=user.id, method="api_key_dev")
            return user

    # Development mode - allow anonymous access
    if settings.is_development:
        logger.warning("anonymous_access", message="No authentication provided (dev mode)")
        return User(
            id="anonymous",
            email="",
            roles=["anonymous"],
            metadata={"token_type": "none", "dev_mode": True},
        )

    # Production - require authentication
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_role(required_roles: list[str]) -> Any:
    """
    Dependency to require specific roles.

    Args:
        required_roles: List of roles, user must have at least one

    Returns:
        Dependency function
    """
    from fastapi import Depends

    async def check_role(user: User = Depends(get_current_user)) -> User:
        if not any(role in user.roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return user

    return check_role


def require_admin(user: User) -> User:
    """Check that user has admin role."""
    if "admin" not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user
