"""Cognito setup for DOVA AgentCore OAuth2 authentication.

Creates and configures:
- User Pool for authentication
- App Client for machine-to-machine auth
- Resource Server for API scopes
- Domain for OAuth2 endpoints
"""

import secrets
from dataclasses import dataclass

import boto3
import structlog
from botocore.exceptions import ClientError

logger = structlog.get_logger(__name__)


@dataclass
class CognitoSetupResult:
    """Result of Cognito setup."""

    success: bool
    user_pool_id: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    domain: str | None = None
    error: str | None = None


class CognitoManager:
    """Manages Cognito resources for DOVA."""

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.cognito = boto3.client("cognito-idp", region_name=region)
        self._logger = logger.bind(component="cognito_manager")

    def setup_cognito(self, stack_name: str) -> CognitoSetupResult:
        """Set up complete Cognito infrastructure for DOVA.

        Creates:
        1. User Pool
        2. Resource Server (for API scopes)
        3. App Client (for machine-to-machine auth)
        4. Domain (for OAuth2 token endpoint)

        Args:
            stack_name: Name of the stack (used for naming resources)

        Returns:
            CognitoSetupResult with credentials and IDs
        """
        self._logger.info("setting_up_cognito", stack_name=stack_name)

        try:
            # Step 1: Create or get User Pool
            user_pool_id = self._create_or_get_user_pool(stack_name)
            if not user_pool_id:
                return CognitoSetupResult(
                    success=False,
                    error="Failed to create user pool",
                )

            # Step 2: Create Resource Server for scopes
            self._create_resource_server(user_pool_id, stack_name)

            # Step 3: Create or get App Client
            client_id, client_secret = self._create_or_get_app_client(
                user_pool_id, stack_name
            )
            if not client_id:
                return CognitoSetupResult(
                    success=False,
                    error="Failed to create app client",
                )

            # Step 4: Create domain
            domain = self._create_or_get_domain(user_pool_id, stack_name)

            self._logger.info(
                "cognito_setup_complete",
                user_pool_id=user_pool_id,
                client_id=client_id,
                domain=domain,
            )

            return CognitoSetupResult(
                success=True,
                user_pool_id=user_pool_id,
                client_id=client_id,
                client_secret=client_secret,
                domain=domain,
            )

        except ClientError as e:
            error_msg = e.response["Error"]["Message"]
            self._logger.error("cognito_setup_failed", error=error_msg)
            return CognitoSetupResult(success=False, error=error_msg)

    def _create_or_get_user_pool(self, stack_name: str) -> str | None:
        """Create or get existing user pool."""
        pool_name = f"{stack_name}-dova-pool"

        # Check if pool already exists
        try:
            response = self.cognito.list_user_pools(MaxResults=60)
            for pool in response.get("UserPools", []):
                if pool["Name"] == pool_name:
                    self._logger.info("user_pool_exists", pool_id=pool["Id"])
                    return pool["Id"]
        except ClientError:
            pass

        # Create new pool
        self._logger.info("creating_user_pool", pool_name=pool_name)

        response = self.cognito.create_user_pool(
            PoolName=pool_name,
            Policies={
                "PasswordPolicy": {
                    "MinimumLength": 12,
                    "RequireUppercase": True,
                    "RequireLowercase": True,
                    "RequireNumbers": True,
                    "RequireSymbols": True,
                }
            },
            AutoVerifiedAttributes=["email"],
            UsernameAttributes=["email"],
            MfaConfiguration="OFF",
            UserPoolTags={
                "Stack": stack_name,
                "ManagedBy": "dova",
            },
        )

        user_pool_id = response["UserPool"]["Id"]
        self._logger.info("user_pool_created", pool_id=user_pool_id)
        return user_pool_id

    def _create_resource_server(self, user_pool_id: str, stack_name: str) -> None:
        """Create resource server for API scopes."""
        identifier = f"{stack_name}-gateway"

        try:
            self.cognito.create_resource_server(
                UserPoolId=user_pool_id,
                Identifier=identifier,
                Name=f"{stack_name} Gateway API",
                Scopes=[
                    {
                        "ScopeName": "read",
                        "ScopeDescription": "Read access to DOVA gateway",
                    },
                    {
                        "ScopeName": "write",
                        "ScopeDescription": "Write access to DOVA gateway",
                    },
                ],
            )
            self._logger.info("resource_server_created", identifier=identifier)
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceExistsException":
                self._logger.info("resource_server_exists", identifier=identifier)
            else:
                raise

    def _create_or_get_app_client(
        self, user_pool_id: str, stack_name: str
    ) -> tuple[str | None, str | None]:
        """Create or get app client for machine-to-machine auth."""
        client_name = f"{stack_name}-machine-client"

        # Check if client already exists
        try:
            response = self.cognito.list_user_pool_clients(
                UserPoolId=user_pool_id, MaxResults=60
            )
            for client in response.get("UserPoolClients", []):
                if client["ClientName"] == client_name:
                    # Get full client details including secret
                    client_details = self.cognito.describe_user_pool_client(
                        UserPoolId=user_pool_id, ClientId=client["ClientId"]
                    )
                    client_secret = client_details["UserPoolClient"].get("ClientSecret")
                    self._logger.info(
                        "app_client_exists",
                        client_id=client["ClientId"],
                        has_secret=bool(client_secret),
                    )
                    return client["ClientId"], client_secret
        except ClientError:
            pass

        # Create new client
        self._logger.info("creating_app_client", client_name=client_name)

        response = self.cognito.create_user_pool_client(
            UserPoolId=user_pool_id,
            ClientName=client_name,
            GenerateSecret=True,
            AllowedOAuthFlows=["client_credentials"],
            AllowedOAuthScopes=[
                f"{stack_name}-gateway/read",
                f"{stack_name}-gateway/write",
            ],
            AllowedOAuthFlowsUserPoolClient=True,
            SupportedIdentityProviders=["COGNITO"],
            ExplicitAuthFlows=[
                "ALLOW_REFRESH_TOKEN_AUTH",
            ],
        )

        client_id = response["UserPoolClient"]["ClientId"]
        client_secret = response["UserPoolClient"].get("ClientSecret")

        self._logger.info("app_client_created", client_id=client_id)
        return client_id, client_secret

    def _create_or_get_domain(self, user_pool_id: str, stack_name: str) -> str | None:
        """Create or get Cognito domain for OAuth2 endpoints."""
        # Generate a unique domain prefix
        domain_prefix = f"{stack_name}-dova-{secrets.token_hex(4)}"

        # Check if domain already exists
        try:
            response = self.cognito.describe_user_pool(UserPoolId=user_pool_id)
            existing_domain = response["UserPool"].get("Domain")
            if existing_domain:
                full_domain = f"{existing_domain}.auth.{self.region}.amazoncognito.com"
                self._logger.info("domain_exists", domain=full_domain)
                return full_domain
        except ClientError:
            pass

        # Create new domain
        self._logger.info("creating_domain", domain_prefix=domain_prefix)

        try:
            self.cognito.create_user_pool_domain(
                Domain=domain_prefix,
                UserPoolId=user_pool_id,
            )
            full_domain = f"{domain_prefix}.auth.{self.region}.amazoncognito.com"
            self._logger.info("domain_created", domain=full_domain)
            return full_domain
        except ClientError as e:
            if e.response["Error"]["Code"] == "InvalidParameterException":
                # Domain might be taken, try with different suffix
                domain_prefix = f"{stack_name}-{secrets.token_hex(6)}"
                self.cognito.create_user_pool_domain(
                    Domain=domain_prefix,
                    UserPoolId=user_pool_id,
                )
                full_domain = f"{domain_prefix}.auth.{self.region}.amazoncognito.com"
                return full_domain
            raise

    def delete_cognito_resources(self, stack_name: str) -> bool:
        """Delete all Cognito resources for a stack."""
        pool_name = f"{stack_name}-dova-pool"

        self._logger.info("deleting_cognito_resources", stack_name=stack_name)

        try:
            # Find the user pool
            response = self.cognito.list_user_pools(MaxResults=60)
            user_pool_id = None
            for pool in response.get("UserPools", []):
                if pool["Name"] == pool_name:
                    user_pool_id = pool["Id"]
                    break

            if not user_pool_id:
                self._logger.info("user_pool_not_found", pool_name=pool_name)
                return True

            # Get pool details for domain
            pool_details = self.cognito.describe_user_pool(UserPoolId=user_pool_id)
            domain = pool_details["UserPool"].get("Domain")

            # Delete domain first (required before deleting pool)
            if domain:
                try:
                    self.cognito.delete_user_pool_domain(
                        Domain=domain,
                        UserPoolId=user_pool_id,
                    )
                    self._logger.info("domain_deleted", domain=domain)
                except ClientError:
                    pass

            # Delete the user pool (cascades to clients and resource servers)
            self.cognito.delete_user_pool(UserPoolId=user_pool_id)
            self._logger.info("user_pool_deleted", pool_id=user_pool_id)

            return True

        except Exception as e:
            self._logger.error("cognito_cleanup_failed", error=str(e))
            return False

    def validate_setup(self, stack_name: str) -> dict:
        """Validate Cognito setup for a stack."""
        pool_name = f"{stack_name}-dova-pool"
        result = {
            "user_pool": False,
            "app_client": False,
            "resource_server": False,
            "domain": False,
        }

        try:
            # Check user pool
            response = self.cognito.list_user_pools(MaxResults=60)
            for pool in response.get("UserPools", []):
                if pool["Name"] == pool_name:
                    user_pool_id = pool["Id"]
                    result["user_pool"] = True

                    # Check app client
                    clients = self.cognito.list_user_pool_clients(
                        UserPoolId=user_pool_id, MaxResults=60
                    )
                    for client in clients.get("UserPoolClients", []):
                        if client["ClientName"] == f"{stack_name}-machine-client":
                            result["app_client"] = True
                            break

                    # Check resource server
                    try:
                        self.cognito.describe_resource_server(
                            UserPoolId=user_pool_id,
                            Identifier=f"{stack_name}-gateway",
                        )
                        result["resource_server"] = True
                    except ClientError:
                        pass

                    # Check domain
                    pool_details = self.cognito.describe_user_pool(UserPoolId=user_pool_id)
                    if pool_details["UserPool"].get("Domain"):
                        result["domain"] = True

                    break

        except Exception as e:
            self._logger.error("validation_failed", error=str(e))

        return result
