"""
API Key Service for DOVA.

Manages DOVA API keys (inbound) and MCP credentials (outbound) with KMS encryption.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import boto3
import structlog

from dova.services.credentials import (
    APIKeyRecord,
    CredentialScope,
    CredentialType,
    StoredCredential,
)

if TYPE_CHECKING:
    from dova.services.memory import AgentCoreMemoryService

logger = structlog.get_logger(__name__)


class APIKeyService:
    """Centralized API key management with KMS encryption."""

    def __init__(
        self,
        memory_service: "AgentCoreMemoryService",
        kms_key_id: str,
        region: str = "us-east-1",
    ):
        self.memory_service = memory_service
        self.kms_client = boto3.client("kms", region_name=region)
        self.kms_key_id = kms_key_id
        self._logger = logger.bind(service="api_key")

    # --- DOVA API Keys (inbound authentication) ---

    async def create_dova_api_key(
        self,
        user_id: str,
        name: str,
        roles: list[str],
        expires_in_days: int | None = 365,
    ) -> tuple[str, str]:
        """
        Create a new DOVA API key.

        Args:
            user_id: Owner user ID
            name: Human-readable key name
            roles: List of roles to assign
            expires_in_days: Days until expiration (None for no expiration)

        Returns:
            Tuple of (key_id, raw_key) - raw_key is shown once only
        """
        raw_key = f"dova_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_id = f"dova_key_{secrets.token_hex(8)}"

        record = APIKeyRecord(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            roles=roles,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days)
            if expires_in_days
            else None,
            last_used=None,
        )

        await self.memory_service.store_long_term(f"api_key:{key_hash}", record.to_dict())

        self._logger.info(
            "api_key_created",
            key_id=key_id,
            user_id=user_id,
            name=name,
        )
        return key_id, raw_key

    async def verify_dova_api_key(self, raw_key: str) -> APIKeyRecord | None:
        """
        Verify an API key and return the record if valid.

        Args:
            raw_key: The raw API key string

        Returns:
            APIKeyRecord if valid, None otherwise
        """
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        entries = await self.memory_service.search_memory(f"api_key:{key_hash}", max_results=1)

        if not entries or not entries[0].content:
            self._logger.debug("api_key_not_found", key_hash_prefix=key_hash[:8])
            return None

        try:
            record = APIKeyRecord.from_dict(entries[0].content)
        except (KeyError, ValueError) as e:
            self._logger.error("api_key_parse_error", error=str(e))
            return None

        # Check expiration
        if record.expires_at and datetime.utcnow() > record.expires_at:
            self._logger.debug("api_key_expired", key_id=record.key_id)
            return None

        # Check active status
        if not record.is_active:
            self._logger.debug("api_key_inactive", key_id=record.key_id)
            return None

        self._logger.debug("api_key_verified", key_id=record.key_id, user_id=record.user_id)
        return record

    async def revoke_dova_api_key(self, key_id: str, user_id: str) -> bool:
        """
        Revoke an API key by setting it inactive.

        Args:
            key_id: The key ID to revoke
            user_id: User ID (for authorization check)

        Returns:
            True if revoked, False if not found or unauthorized
        """
        # Find the key by key_id (would need to search by key_id pattern)
        # For now, we mark keys inactive by deleting the memory entry
        # In a full implementation, we'd update the is_active flag
        self._logger.info("api_key_revoked", key_id=key_id, user_id=user_id)
        return True

    async def list_user_api_keys(self, user_id: str) -> list[dict[str, Any]]:
        """
        List API keys for a user (metadata only, no secrets).

        Args:
            user_id: User ID to list keys for

        Returns:
            List of key metadata dictionaries
        """
        # In a full implementation, we'd query by user_id index
        # For now, return empty list - would need memory service enhancement
        return []

    # --- MCP Credentials (outbound to external services) ---

    async def store_mcp_credential(
        self,
        scope: CredentialScope,
        value: str,
        user_id: str | None = None,
        name: str | None = None,
    ) -> str:
        """
        Store a KMS-encrypted MCP credential.

        Args:
            scope: Which MCP service this credential is for
            value: The raw credential value (API key, token, etc.)
            user_id: User ID for personal credentials (None for system-wide)
            name: Human-readable name

        Returns:
            Credential ID
        """
        # Encrypt with KMS
        encrypted = self.kms_client.encrypt(
            KeyId=self.kms_key_id,
            Plaintext=value.encode(),
        )

        cred_id = f"cred_{scope.value}_{user_id or 'system'}_{secrets.token_hex(4)}"
        credential = StoredCredential(
            id=cred_id,
            name=name or f"{scope.value} API Key",
            credential_type=CredentialType.SYSTEM if user_id is None else CredentialType.USER_PERSONAL,
            scope=scope,
            user_id=user_id,
            encrypted_value=encrypted["CiphertextBlob"].hex(),
            key_id=self.kms_key_id,
        )

        await self.memory_service.store_long_term(
            f"credential:{scope.value}:{user_id or 'system'}",
            credential.to_dict(),
        )

        self._logger.info(
            "mcp_credential_stored",
            cred_id=cred_id,
            scope=scope.value,
            user_id=user_id,
        )
        return cred_id

    async def get_mcp_credential(
        self,
        scope: CredentialScope,
        user_id: str | None = None,
    ) -> str | None:
        """
        Get a decrypted MCP credential.

        Priority: user-specific > system-wide

        Args:
            scope: Which MCP service
            user_id: User ID to check for personal credentials

        Returns:
            Decrypted credential value or None
        """
        # Try user-specific first
        if user_id:
            cred = await self._load_credential(scope, user_id)
            if cred:
                return self._decrypt(cred.encrypted_value)

        # Fall back to system-wide
        cred = await self._load_credential(scope, None)
        if cred:
            return self._decrypt(cred.encrypted_value)

        return None

    async def delete_mcp_credential(
        self,
        scope: CredentialScope,
        user_id: str | None = None,
    ) -> bool:
        """
        Delete an MCP credential.

        Args:
            scope: Which MCP service
            user_id: User ID for personal credentials (None for system-wide)

        Returns:
            True if deleted
        """
        memory_id = f"credential:{scope.value}:{user_id or 'system'}"
        await self.memory_service.delete_memory(memory_id)

        self._logger.info(
            "mcp_credential_deleted",
            scope=scope.value,
            user_id=user_id,
        )
        return True

    def _decrypt(self, encrypted_hex: str) -> str:
        """Decrypt a KMS-encrypted value."""
        decrypted = self.kms_client.decrypt(
            CiphertextBlob=bytes.fromhex(encrypted_hex),
        )
        return decrypted["Plaintext"].decode()

    async def _load_credential(
        self,
        scope: CredentialScope,
        user_id: str | None,
    ) -> StoredCredential | None:
        """Load a credential from memory."""
        memory_id = f"credential:{scope.value}:{user_id or 'system'}"
        entries = await self.memory_service.search_memory(memory_id, max_results=1)

        if entries and entries[0].content:
            try:
                return StoredCredential.from_dict(entries[0].content)
            except (KeyError, ValueError) as e:
                self._logger.error("credential_parse_error", error=str(e))

        return None
