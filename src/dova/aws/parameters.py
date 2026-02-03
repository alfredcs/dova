"""SSM Parameter Store and Secrets Manager setup for DOVA.

Stores configuration and credentials:
- SSM: Public configuration (Cognito domain, client ID, gateway URL)
- Secrets Manager: Sensitive data (client secrets)
"""

from dataclasses import dataclass

import boto3
import structlog
from botocore.exceptions import ClientError

logger = structlog.get_logger(__name__)


@dataclass
class ParameterSetupResult:
    """Result of parameter setup."""

    success: bool
    parameters_created: list[str] | None = None
    secrets_created: list[str] | None = None
    error: str | None = None


class ParameterManager:
    """Manages SSM parameters and secrets for DOVA."""

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.ssm = boto3.client("ssm", region_name=region)
        self.secrets = boto3.client("secretsmanager", region_name=region)
        self._logger = logger.bind(component="parameter_manager")

    def store_configuration(
        self,
        stack_name: str,
        cognito_domain: str,
        client_id: str,
        client_secret: str,
        gateway_url: str | None = None,
        memory_id: str | None = None,
    ) -> ParameterSetupResult:
        """Store all DOVA configuration in SSM and Secrets Manager.

        Args:
            stack_name: Stack name for parameter paths
            cognito_domain: Cognito domain for OAuth2
            client_id: Cognito app client ID
            client_secret: Cognito app client secret
            gateway_url: AgentCore Gateway URL (optional)
            memory_id: AgentCore Memory ID (optional)

        Returns:
            ParameterSetupResult with created resource names
        """
        self._logger.info("storing_configuration", stack_name=stack_name)

        parameters_created = []
        secrets_created = []

        try:
            # Store SSM parameters
            ssm_params = {
                f"/{stack_name}/cognito_provider": cognito_domain,
                f"/{stack_name}/machine_client_id": client_id,
            }

            if gateway_url:
                ssm_params[f"/{stack_name}/gateway_url"] = gateway_url

            if memory_id:
                ssm_params[f"/{stack_name}/memory_id"] = memory_id

            for param_name, param_value in ssm_params.items():
                self._put_parameter(param_name, param_value)
                parameters_created.append(param_name)

            # Store secret
            secret_name = f"/{stack_name}/machine_client_secret"
            self._put_secret(secret_name, client_secret)
            secrets_created.append(secret_name)

            self._logger.info(
                "configuration_stored",
                parameters=len(parameters_created),
                secrets=len(secrets_created),
            )

            return ParameterSetupResult(
                success=True,
                parameters_created=parameters_created,
                secrets_created=secrets_created,
            )

        except ClientError as e:
            error_msg = e.response["Error"]["Message"]
            self._logger.error("configuration_storage_failed", error=error_msg)
            return ParameterSetupResult(success=False, error=error_msg)

    def _put_parameter(self, name: str, value: str, secure: bool = False) -> None:
        """Put a parameter in SSM Parameter Store."""
        param_type = "SecureString" if secure else "String"

        try:
            self.ssm.put_parameter(
                Name=name,
                Value=value,
                Type=param_type,
                Overwrite=True,
                Tags=[{"Key": "ManagedBy", "Value": "dova"}],
            )
            self._logger.debug("parameter_stored", name=name)
        except ClientError as e:
            if e.response["Error"]["Code"] == "ParameterAlreadyExists":
                # Update existing parameter
                self.ssm.put_parameter(
                    Name=name,
                    Value=value,
                    Type=param_type,
                    Overwrite=True,
                )
                self._logger.debug("parameter_updated", name=name)
            else:
                raise

    def _put_secret(self, name: str, value: str) -> None:
        """Put a secret in Secrets Manager."""
        try:
            self.secrets.create_secret(
                Name=name,
                SecretString=value,
                Tags=[{"Key": "ManagedBy", "Value": "dova"}],
            )
            self._logger.debug("secret_created", name=name)
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceExistsException":
                # Update existing secret
                self.secrets.update_secret(
                    SecretId=name,
                    SecretString=value,
                )
                self._logger.debug("secret_updated", name=name)
            else:
                raise

    def get_parameter(self, name: str) -> str | None:
        """Get a parameter from SSM."""
        try:
            response = self.ssm.get_parameter(Name=name, WithDecryption=True)
            return response["Parameter"]["Value"]
        except ClientError:
            return None

    def get_secret(self, name: str) -> str | None:
        """Get a secret from Secrets Manager."""
        try:
            response = self.secrets.get_secret_value(SecretId=name)
            return response["SecretString"]
        except ClientError:
            return None

    def delete_configuration(self, stack_name: str) -> bool:
        """Delete all DOVA configuration for a stack."""
        self._logger.info("deleting_configuration", stack_name=stack_name)

        try:
            # Delete SSM parameters
            param_names = [
                f"/{stack_name}/cognito_provider",
                f"/{stack_name}/machine_client_id",
                f"/{stack_name}/gateway_url",
                f"/{stack_name}/memory_id",
            ]

            for param_name in param_names:
                try:
                    self.ssm.delete_parameter(Name=param_name)
                    self._logger.debug("parameter_deleted", name=param_name)
                except ClientError as e:
                    if e.response["Error"]["Code"] != "ParameterNotFound":
                        raise

            # Delete secret
            secret_name = f"/{stack_name}/machine_client_secret"
            try:
                self.secrets.delete_secret(
                    SecretId=secret_name,
                    ForceDeleteWithoutRecovery=True,
                )
                self._logger.debug("secret_deleted", name=secret_name)
            except ClientError as e:
                if e.response["Error"]["Code"] != "ResourceNotFoundException":
                    raise

            return True

        except Exception as e:
            self._logger.error("configuration_cleanup_failed", error=str(e))
            return False

    def validate_configuration(self, stack_name: str) -> dict:
        """Validate stored configuration for a stack."""
        result = {
            "cognito_provider": False,
            "machine_client_id": False,
            "machine_client_secret": False,
            "gateway_url": False,
            "memory_id": False,
        }

        param_mapping = {
            "cognito_provider": f"/{stack_name}/cognito_provider",
            "machine_client_id": f"/{stack_name}/machine_client_id",
            "gateway_url": f"/{stack_name}/gateway_url",
            "memory_id": f"/{stack_name}/memory_id",
        }

        for key, param_name in param_mapping.items():
            if self.get_parameter(param_name):
                result[key] = True

        # Check secret
        if self.get_secret(f"/{stack_name}/machine_client_secret"):
            result["machine_client_secret"] = True

        return result

    def generate_env_file(self, stack_name: str, output_path: str = ".env") -> bool:
        """Generate .env file from stored configuration.

        Args:
            stack_name: Stack name to fetch configuration for
            output_path: Path to write .env file

        Returns:
            True if successful
        """
        self._logger.info("generating_env_file", stack_name=stack_name, output=output_path)

        try:
            # Fetch all parameters
            cognito_provider = self.get_parameter(f"/{stack_name}/cognito_provider")
            client_id = self.get_parameter(f"/{stack_name}/machine_client_id")
            gateway_url = self.get_parameter(f"/{stack_name}/gateway_url")
            memory_id = self.get_parameter(f"/{stack_name}/memory_id")

            # Fetch secret
            client_secret = self.get_secret(f"/{stack_name}/machine_client_secret")

            # Build env content
            lines = [
                "# DOVA AWS Configuration",
                f"# Generated for stack: {stack_name}",
                "",
                f"STACK_NAME={stack_name}",
                f"AWS_REGION={self.region}",
                "",
                "# AgentCore Settings",
                f"AGENTCORE_STACK_NAME={stack_name}",
                "AGENTCORE_RUNTIME_MODE=agentcore",
            ]

            if cognito_provider:
                lines.append("# Cognito OAuth2")
                lines.append(f"COGNITO_DOMAIN={cognito_provider}")

            if client_id:
                lines.append(f"COGNITO_CLIENT_ID={client_id}")

            if client_secret:
                lines.append(f"COGNITO_CLIENT_SECRET={client_secret}")

            if gateway_url:
                lines.append(f"AGENTCORE_GATEWAY_URL={gateway_url}")

            if memory_id:
                lines.append(f"AGENTCORE_MEMORY_ID={memory_id}")

            # Write file
            with open(output_path, "w") as f:
                f.write("\n".join(lines) + "\n")

            self._logger.info("env_file_generated", path=output_path)
            return True

        except Exception as e:
            self._logger.error("env_file_generation_failed", error=str(e))
            return False
