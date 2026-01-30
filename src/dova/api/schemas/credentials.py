"""
Credentials API Schemas for DOVA.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from dova.services.credentials import CredentialScope


class CreateAPIKeyRequest(BaseModel):
    """Request to create a new DOVA API key."""

    name: str = Field(..., description="Human-readable name for the API key", max_length=100)
    roles: list[str] = Field(
        default=["api_user"],
        description="Roles to assign to this API key",
    )
    expires_in_days: int | None = Field(
        default=365,
        description="Days until expiration (null for no expiration)",
        ge=1,
        le=3650,
    )


class CreateAPIKeyResponse(BaseModel):
    """Response after creating an API key."""

    key_id: str = Field(..., description="Unique key identifier")
    api_key: str = Field(..., description="The API key (shown once only)")
    name: str = Field(..., description="Key name")
    roles: list[str] = Field(..., description="Assigned roles")
    expires_at: datetime | None = Field(None, description="Expiration timestamp")


class APIKeyInfo(BaseModel):
    """API key metadata (no secret)."""

    key_id: str = Field(..., description="Unique key identifier")
    name: str = Field(..., description="Key name")
    roles: list[str] = Field(..., description="Assigned roles")
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: datetime | None = Field(None, description="Expiration timestamp")
    last_used: datetime | None = Field(None, description="Last usage timestamp")
    is_active: bool = Field(..., description="Whether the key is active")


class ListAPIKeysResponse(BaseModel):
    """Response listing user's API keys."""

    keys: list[APIKeyInfo] = Field(default_factory=list, description="API key metadata")


class StoreMCPCredentialRequest(BaseModel):
    """Request to store an MCP credential."""

    scope: CredentialScope = Field(..., description="Which MCP service this credential is for")
    api_key: str = Field(..., description="The API key or token value", min_length=1)
    name: str | None = Field(None, description="Optional human-readable name", max_length=100)


class StoreMCPCredentialResponse(BaseModel):
    """Response after storing an MCP credential."""

    credential_id: str = Field(..., description="Unique credential identifier")
    scope: CredentialScope = Field(..., description="Credential scope")
    name: str = Field(..., description="Credential name")


class MCPCredentialInfo(BaseModel):
    """MCP credential metadata (no secret)."""

    credential_id: str = Field(..., description="Unique credential identifier")
    scope: CredentialScope = Field(..., description="Credential scope")
    name: str = Field(..., description="Credential name")
    created_at: datetime = Field(..., description="Creation timestamp")
    is_system: bool = Field(..., description="Whether this is a system-wide credential")


class DeleteCredentialResponse(BaseModel):
    """Response after deleting a credential."""

    success: bool = Field(..., description="Whether deletion was successful")
    message: str = Field(..., description="Status message")
