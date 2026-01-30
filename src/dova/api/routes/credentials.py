"""
Credentials API Routes for DOVA.

CRUD operations for API keys and MCP credentials.
"""

from datetime import datetime, timedelta

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status

from dova.api.middleware.auth import User, get_current_user, require_admin
from dova.api.schemas.credentials import (
    APIKeyInfo,
    CreateAPIKeyRequest,
    CreateAPIKeyResponse,
    DeleteCredentialResponse,
    ListAPIKeysResponse,
    MCPCredentialInfo,
    StoreMCPCredentialRequest,
    StoreMCPCredentialResponse,
)
from dova.config.settings import get_settings
from dova.services.credentials import CredentialScope

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/credentials")


def _get_api_key_service(request: Request):
    """Get API key service from app state."""
    service = getattr(request.app.state, "api_key_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key service not configured",
        )
    return service


# --- DOVA API Keys ---


@router.post("/api-keys", response_model=CreateAPIKeyResponse)
async def create_api_key(
    request: Request,
    body: CreateAPIKeyRequest,
    user: User = Depends(get_current_user),
) -> CreateAPIKeyResponse:
    """
    Create a new DOVA API key.

    The API key is returned once and cannot be retrieved later.
    Store it securely.
    """
    service = _get_api_key_service(request)
    settings = get_settings()

    key_id, raw_key = await service.create_dova_api_key(
        user_id=user.id,
        name=body.name,
        roles=body.roles,
        expires_in_days=body.expires_in_days or settings.auth.api_key_expiry_days,
    )

    expires_at = None
    if body.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=body.expires_in_days)

    logger.info("api_key_created", key_id=key_id, user_id=user.id)

    return CreateAPIKeyResponse(
        key_id=key_id,
        api_key=raw_key,
        name=body.name,
        roles=body.roles,
        expires_at=expires_at,
    )


@router.get("/api-keys", response_model=ListAPIKeysResponse)
async def list_api_keys(
    request: Request,
    user: User = Depends(get_current_user),
) -> ListAPIKeysResponse:
    """List the current user's API keys (metadata only, no secrets)."""
    service = _get_api_key_service(request)

    keys = await service.list_user_api_keys(user.id)

    return ListAPIKeysResponse(
        keys=[
            APIKeyInfo(
                key_id=k["key_id"],
                name=k["name"],
                roles=k["roles"],
                created_at=datetime.fromisoformat(k["created_at"]),
                expires_at=datetime.fromisoformat(k["expires_at"]) if k.get("expires_at") else None,
                last_used=datetime.fromisoformat(k["last_used"]) if k.get("last_used") else None,
                is_active=k.get("is_active", True),
            )
            for k in keys
        ]
    )


@router.delete("/api-keys/{key_id}", response_model=DeleteCredentialResponse)
async def revoke_api_key(
    request: Request,
    key_id: str,
    user: User = Depends(get_current_user),
) -> DeleteCredentialResponse:
    """Revoke an API key."""
    service = _get_api_key_service(request)

    success = await service.revoke_dova_api_key(key_id, user.id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found or not owned by user",
        )

    logger.info("api_key_revoked", key_id=key_id, user_id=user.id)

    return DeleteCredentialResponse(
        success=True,
        message=f"API key {key_id} has been revoked",
    )


# --- MCP Credentials (user-specific) ---


@router.post("/mcp", response_model=StoreMCPCredentialResponse)
async def store_mcp_credential(
    request: Request,
    body: StoreMCPCredentialRequest,
    user: User = Depends(get_current_user),
) -> StoreMCPCredentialResponse:
    """
    Store a personal MCP credential (e.g., GitHub token, HuggingFace API key).

    The credential is encrypted with KMS and associated with your user account.
    Personal credentials take priority over system-wide credentials.
    """
    service = _get_api_key_service(request)

    cred_id = await service.store_mcp_credential(
        scope=body.scope,
        value=body.api_key,
        user_id=user.id,
        name=body.name,
    )

    logger.info("mcp_credential_stored", cred_id=cred_id, scope=body.scope.value, user_id=user.id)

    return StoreMCPCredentialResponse(
        credential_id=cred_id,
        scope=body.scope,
        name=body.name or f"{body.scope.value} API Key",
    )


@router.delete("/mcp/{scope}", response_model=DeleteCredentialResponse)
async def delete_mcp_credential(
    request: Request,
    scope: CredentialScope,
    user: User = Depends(get_current_user),
) -> DeleteCredentialResponse:
    """Delete a personal MCP credential."""
    service = _get_api_key_service(request)

    await service.delete_mcp_credential(scope=scope, user_id=user.id)

    logger.info("mcp_credential_deleted", scope=scope.value, user_id=user.id)

    return DeleteCredentialResponse(
        success=True,
        message=f"MCP credential for {scope.value} has been deleted",
    )


# --- MCP Credentials (system-wide, admin only) ---


@router.post("/mcp/system", response_model=StoreMCPCredentialResponse)
async def store_system_mcp_credential(
    request: Request,
    body: StoreMCPCredentialRequest,
    user: User = Depends(get_current_user),
) -> StoreMCPCredentialResponse:
    """
    Store a system-wide MCP credential (admin only).

    System credentials are used as fallback when users don't have personal credentials.
    """
    require_admin(user)

    service = _get_api_key_service(request)

    cred_id = await service.store_mcp_credential(
        scope=body.scope,
        value=body.api_key,
        user_id=None,  # System-wide
        name=body.name,
    )

    logger.info("system_mcp_credential_stored", cred_id=cred_id, scope=body.scope.value)

    return StoreMCPCredentialResponse(
        credential_id=cred_id,
        scope=body.scope,
        name=body.name or f"{body.scope.value} System Key",
    )


@router.delete("/mcp/system/{scope}", response_model=DeleteCredentialResponse)
async def delete_system_mcp_credential(
    request: Request,
    scope: CredentialScope,
    user: User = Depends(get_current_user),
) -> DeleteCredentialResponse:
    """Delete a system-wide MCP credential (admin only)."""
    require_admin(user)

    service = _get_api_key_service(request)

    await service.delete_mcp_credential(scope=scope, user_id=None)

    logger.info("system_mcp_credential_deleted", scope=scope.value)

    return DeleteCredentialResponse(
        success=True,
        message=f"System MCP credential for {scope.value} has been deleted",
    )
