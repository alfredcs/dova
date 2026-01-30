"""
Cognito JWT Verifier for DOVA.

Verifies JWT tokens using Cognito JWKS with RS256 signature validation.
"""

import time
from dataclasses import dataclass
from typing import Any

import jwt
import structlog
from jwt import PyJWKClient

logger = structlog.get_logger(__name__)


@dataclass
class JWTVerifierConfig:
    """Configuration for JWT verification."""

    region: str
    user_pool_id: str
    client_id: str
    jwks_cache_ttl: int = 3600  # 1 hour


class CognitoJWTVerifier:
    """Verify Cognito JWT tokens with JWKS validation."""

    def __init__(self, config: JWTVerifierConfig):
        self.config = config
        self._jwks_client: PyJWKClient | None = None
        self._jwks_last_refresh: float = 0

    @property
    def issuer(self) -> str:
        """Get the expected token issuer URL."""
        return (
            f"https://cognito-idp.{self.config.region}.amazonaws.com/{self.config.user_pool_id}"
        )

    @property
    def jwks_url(self) -> str:
        """Get the JWKS endpoint URL."""
        return f"{self.issuer}/.well-known/jwks.json"

    def _get_jwks_client(self) -> PyJWKClient:
        """Get or refresh the JWKS client."""
        now = time.time()
        if (
            self._jwks_client is None
            or now - self._jwks_last_refresh > self.config.jwks_cache_ttl
        ):
            self._jwks_client = PyJWKClient(self.jwks_url, cache_keys=True)
            self._jwks_last_refresh = now
            logger.debug("jwks_client_refreshed", jwks_url=self.jwks_url)
        return self._jwks_client

    async def verify(self, token: str) -> dict[str, Any] | None:
        """
        Verify JWT and return claims if valid.

        Args:
            token: The JWT token string

        Returns:
            Claims dictionary if valid, None otherwise
        """
        try:
            jwks_client = self._get_jwks_client()
            signing_key = jwks_client.get_signing_key_from_jwt(token)

            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self.config.client_id,
                issuer=self.issuer,
            )

            # Validate token_use claim
            token_use = claims.get("token_use")
            if token_use not in ("access", "id"):
                logger.warning("jwt_invalid_token_use", token_use=token_use)
                return None

            logger.debug(
                "jwt_verified",
                sub=claims.get("sub"),
                token_use=token_use,
            )
            return claims

        except jwt.ExpiredSignatureError:
            logger.warning("jwt_expired")
            return None
        except jwt.InvalidAudienceError:
            logger.warning("jwt_invalid_audience")
            return None
        except jwt.InvalidIssuerError:
            logger.warning("jwt_invalid_issuer")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("jwt_invalid_token", error=str(e))
            return None
        except Exception as e:
            logger.error("jwt_verification_error", error=str(e))
            return None
