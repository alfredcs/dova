"""
Credential Data Models for DOVA.

Defines types for stored credentials (MCP API keys) and DOVA API keys.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class CredentialType(Enum):
    """Type of credential ownership."""

    SYSTEM = "system"  # System-wide keys (fallback)
    USER_PERSONAL = "personal"  # Per-user API keys


class CredentialScope(Enum):
    """Scope defining which service a credential is for."""

    MCP_GITHUB = "mcp_github"
    MCP_HUGGINGFACE = "mcp_huggingface"
    MCP_TAVILY = "mcp_tavily"
    CUSTOM_SOURCE = "custom_source"


@dataclass
class StoredCredential:
    """KMS-encrypted credential stored in memory service."""

    id: str
    name: str
    credential_type: CredentialType
    scope: CredentialScope
    user_id: str | None  # None for system-wide
    encrypted_value: str  # KMS-encrypted hex
    key_id: str  # KMS key ID
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    last_rotated: datetime | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "credential_type": self.credential_type.value,
            "scope": self.scope.value,
            "user_id": self.user_id,
            "encrypted_value": self.encrypted_value,
            "key_id": self.key_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_rotated": self.last_rotated.isoformat() if self.last_rotated else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StoredCredential":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            credential_type=CredentialType(data["credential_type"]),
            scope=CredentialScope(data["scope"]),
            user_id=data.get("user_id"),
            encrypted_value=data["encrypted_value"],
            key_id=data["key_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            last_rotated=datetime.fromisoformat(data["last_rotated"])
            if data.get("last_rotated")
            else None,
        )


@dataclass
class APIKeyRecord:
    """DOVA API key for programmatic access."""

    key_id: str
    key_hash: str  # SHA256 for lookup
    user_id: str
    name: str
    roles: list[str]
    created_at: datetime
    expires_at: datetime | None
    last_used: datetime | None
    is_active: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "key_id": self.key_id,
            "key_hash": self.key_hash,
            "user_id": self.user_id,
            "name": self.name,
            "roles": self.roles,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "APIKeyRecord":
        """Create from dictionary."""
        return cls(
            key_id=data["key_id"],
            key_hash=data["key_hash"],
            user_id=data["user_id"],
            name=data["name"],
            roles=data["roles"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            is_active=data.get("is_active", True),
        )
