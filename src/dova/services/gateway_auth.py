"""OAuth2 Client Credentials flow for AgentCore Gateway.

Provides authentication for accessing AgentCore Gateway endpoints
using OAuth2 client credentials flow with AWS Cognito.
"""

import base64
import os
import time
from dataclasses import dataclass

import httpx
import structlog

from dova.services.aws_config import get_secret, get_ssm_parameter

logger = structlog.get_logger(__name__)


@dataclass
class GatewayToken:
    """Represents an OAuth2 access token."""

    access_token: str
    token_type: str
    expires_in: int
    expires_at: float  # Unix timestamp


# Cache for gateway tokens
_token_cache: dict[str, GatewayToken] = {}


def get_stack_name() -> str:
    """Get the CloudFormation stack name from environment."""
    stack_name = os.environ.get("STACK_NAME")
    if not stack_name:
        raise ValueError("STACK_NAME environment variable is required for gateway auth")
    return stack_name


def get_gateway_access_token(force_refresh: bool = False) -> str:
    """Get OAuth2 access token using client credentials flow.

    Args:
        force_refresh: Force token refresh even if cached token is valid

    Returns:
        Access token string

    Raises:
        ValueError: If required environment variables not set
        httpx.HTTPStatusError: If token request fails
    """
    stack_name = get_stack_name()
    cache_key = f"gateway_token:{stack_name}"

    # Check cache
    if not force_refresh and cache_key in _token_cache:
        cached_token = _token_cache[cache_key]
        # Check if token is still valid (with 60s buffer)
        if time.time() < cached_token.expires_at - 60:
            logger.debug("gateway_token_cache_hit", stack=stack_name)
            return cached_token.access_token

    # Get Cognito config from SSM/Secrets Manager
    cognito_domain = get_ssm_parameter(f"/{stack_name}/cognito_provider")
    client_id = get_ssm_parameter(f"/{stack_name}/machine_client_id")
    client_secret = get_secret(f"/{stack_name}/machine_client_secret")

    # Build token URL
    token_url = f"https://{cognito_domain}/oauth2/token"

    # Build authorization header (Basic auth with client credentials)
    credentials = f"{client_id}:{client_secret}"
    b64_credentials = base64.b64encode(credentials.encode()).decode()

    # Build scopes
    scopes = f"{stack_name}-gateway/read {stack_name}-gateway/write"

    logger.debug(
        "gateway_token_request",
        stack=stack_name,
        token_url=token_url,
        scopes=scopes,
    )

    # Make token request
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            token_url,
            headers={
                "Authorization": f"Basic {b64_credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "client_credentials",
                "scope": scopes,
            },
        )

        response.raise_for_status()
        token_data = response.json()

    # Parse response
    access_token = token_data["access_token"]
    token_type = token_data.get("token_type", "Bearer")
    expires_in = token_data.get("expires_in", 3600)

    # Cache token
    _token_cache[cache_key] = GatewayToken(
        access_token=access_token,
        token_type=token_type,
        expires_in=expires_in,
        expires_at=time.time() + expires_in,
    )

    logger.info(
        "gateway_token_acquired",
        stack=stack_name,
        expires_in=expires_in,
    )

    return access_token


async def get_gateway_access_token_async(force_refresh: bool = False) -> str:
    """Async version of get_gateway_access_token.

    Args:
        force_refresh: Force token refresh even if cached token is valid

    Returns:
        Access token string
    """
    stack_name = get_stack_name()
    cache_key = f"gateway_token:{stack_name}"

    # Check cache
    if not force_refresh and cache_key in _token_cache:
        cached_token = _token_cache[cache_key]
        if time.time() < cached_token.expires_at - 60:
            logger.debug("gateway_token_cache_hit", stack=stack_name)
            return cached_token.access_token

    # Get Cognito config from SSM/Secrets Manager
    cognito_domain = get_ssm_parameter(f"/{stack_name}/cognito_provider")
    client_id = get_ssm_parameter(f"/{stack_name}/machine_client_id")
    client_secret = get_secret(f"/{stack_name}/machine_client_secret")

    token_url = f"https://{cognito_domain}/oauth2/token"
    credentials = f"{client_id}:{client_secret}"
    b64_credentials = base64.b64encode(credentials.encode()).decode()
    scopes = f"{stack_name}-gateway/read {stack_name}-gateway/write"

    logger.debug(
        "gateway_token_request_async",
        stack=stack_name,
        token_url=token_url,
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            token_url,
            headers={
                "Authorization": f"Basic {b64_credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "client_credentials",
                "scope": scopes,
            },
        )

        response.raise_for_status()
        token_data = response.json()

    access_token = token_data["access_token"]
    token_type = token_data.get("token_type", "Bearer")
    expires_in = token_data.get("expires_in", 3600)

    _token_cache[cache_key] = GatewayToken(
        access_token=access_token,
        token_type=token_type,
        expires_in=expires_in,
        expires_at=time.time() + expires_in,
    )

    logger.info(
        "gateway_token_acquired_async",
        stack=stack_name,
        expires_in=expires_in,
    )

    return access_token


def clear_token_cache() -> None:
    """Clear the gateway token cache."""
    _token_cache.clear()
    logger.debug("gateway_token_cache_cleared")


def get_gateway_url() -> str:
    """Get the AgentCore Gateway URL from SSM.

    Returns:
        Gateway URL string
    """
    stack_name = get_stack_name()
    return get_ssm_parameter(f"/{stack_name}/gateway_url")
