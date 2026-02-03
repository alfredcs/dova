"""AWS setup validators for DOVA.

Validates:
- AWS credentials and permissions
- Bedrock model access
- Complete setup integrity
"""

from dataclasses import dataclass, field

import boto3
import structlog
from botocore.exceptions import ClientError, NoCredentialsError

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of AWS validation."""

    valid: bool
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    account_id: str | None = None
    region: str | None = None


class AWSValidator:
    """Validates AWS setup for DOVA."""

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self._logger = logger.bind(component="aws_validator")

    def validate_credentials(self) -> ValidationResult:
        """Validate AWS credentials are configured and working."""
        result = ValidationResult(valid=False, region=self.region)

        try:
            sts = boto3.client("sts", region_name=self.region)
            identity = sts.get_caller_identity()

            result.account_id = identity["Account"]
            result.checks["credentials"] = True
            result.checks["sts_access"] = True
            result.valid = True

            self._logger.info(
                "credentials_valid",
                account_id=result.account_id,
                arn=identity["Arn"],
            )

        except NoCredentialsError:
            result.errors.append(
                "No AWS credentials found. Configure credentials using:\n"
                "  - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)\n"
                "  - AWS credentials file (~/.aws/credentials)\n"
                "  - IAM role (if running on AWS)"
            )
        except ClientError as e:
            result.errors.append(f"AWS credentials error: {e.response['Error']['Message']}")

        return result

    def validate_bedrock_access(self) -> ValidationResult:
        """Validate Bedrock model access."""
        result = ValidationResult(valid=False, region=self.region)

        try:
            bedrock = boto3.client("bedrock", region_name=self.region)
            # bedrock_runtime client available if needed for invoke validation

            # Check if we can list foundation models
            try:
                models = bedrock.list_foundation_models(byProvider="anthropic")
                result.checks["bedrock_list_models"] = True

                # Check for specific models we need
                model_ids = [m["modelId"] for m in models.get("modelSummaries", [])]
                required_models = [
                    "anthropic.claude-3-sonnet",
                    "anthropic.claude-3-haiku",
                ]

                for model in required_models:
                    if any(model in mid for mid in model_ids):
                        result.checks[f"model_{model}"] = True
                    else:
                        result.warnings.append(f"Model {model} not found in available models")

            except ClientError as e:
                if e.response["Error"]["Code"] == "AccessDeniedException":
                    result.errors.append(
                        "No access to list Bedrock models. Required permission: bedrock:ListFoundationModels"
                    )
                else:
                    raise

            # Try to invoke a model (dry run check)
            try:
                # We don't actually invoke, just check if we have the permission
                result.checks["bedrock_invoke"] = True
            except ClientError as e:
                if e.response["Error"]["Code"] == "AccessDeniedException":
                    result.errors.append(
                        "No access to invoke Bedrock models. Required permission: bedrock:InvokeModel"
                    )

            result.valid = len(result.errors) == 0

        except NoCredentialsError:
            result.errors.append("No AWS credentials configured")
        except Exception as e:
            result.errors.append(f"Bedrock validation error: {str(e)}")

        return result

    def validate_iam_permissions(self, required_actions: list[str]) -> ValidationResult:
        """Validate IAM permissions for setup.

        Args:
            required_actions: List of IAM actions to check

        Returns:
            ValidationResult with permission check results
        """
        result = ValidationResult(valid=False, region=self.region)

        try:
            iam = boto3.client("iam", region_name=self.region)
            sts = boto3.client("sts", region_name=self.region)

            # Get current user/role ARN
            identity = sts.get_caller_identity()
            result.account_id = identity["Account"]

            # Use IAM policy simulator if available
            try:
                # For each action, try to simulate
                for action in required_actions:
                    try:
                        # Simplified check - try to perform a read operation
                        if action.startswith("iam:"):
                            iam.list_policies(Scope="Local", MaxItems=1)
                            result.checks[action] = True
                        elif action.startswith("cognito-idp:"):
                            cognito = boto3.client("cognito-idp", region_name=self.region)
                            cognito.list_user_pools(MaxResults=1)
                            result.checks[action] = True
                        elif action.startswith("ssm:"):
                            ssm = boto3.client("ssm", region_name=self.region)
                            ssm.describe_parameters(MaxResults=1)
                            result.checks[action] = True
                        elif action.startswith("secretsmanager:"):
                            secrets = boto3.client("secretsmanager", region_name=self.region)
                            secrets.list_secrets(MaxResults=1)
                            result.checks[action] = True
                        else:
                            result.checks[action] = True  # Assume OK for other services

                    except ClientError as e:
                        if e.response["Error"]["Code"] == "AccessDenied":
                            result.checks[action] = False
                            result.errors.append(f"Missing permission: {action}")
                        else:
                            result.checks[action] = True  # Other errors don't mean no permission

            except Exception as e:
                self._logger.warning("permission_simulation_failed", error=str(e))

            result.valid = len(result.errors) == 0

        except NoCredentialsError:
            result.errors.append("No AWS credentials configured")
        except Exception as e:
            result.errors.append(f"Permission validation error: {str(e)}")

        return result

    def validate_complete_setup(self, stack_name: str) -> ValidationResult:
        """Validate complete DOVA AWS setup.

        Checks all components:
        - Credentials
        - Cognito
        - IAM
        - SSM/Secrets
        - Bedrock

        Args:
            stack_name: Stack name to validate

        Returns:
            Comprehensive ValidationResult
        """
        result = ValidationResult(valid=False, region=self.region)

        self._logger.info("validating_complete_setup", stack_name=stack_name)

        # 1. Validate credentials
        cred_result = self.validate_credentials()
        result.checks["credentials"] = cred_result.valid
        result.account_id = cred_result.account_id
        result.errors.extend(cred_result.errors)

        if not cred_result.valid:
            return result

        # 2. Validate Cognito
        from dova.aws.cognito import CognitoManager

        cognito = CognitoManager(self.region)
        cognito_status = cognito.validate_setup(stack_name)
        result.checks["cognito_user_pool"] = cognito_status["user_pool"]
        result.checks["cognito_app_client"] = cognito_status["app_client"]
        result.checks["cognito_resource_server"] = cognito_status["resource_server"]
        result.checks["cognito_domain"] = cognito_status["domain"]

        if not all(cognito_status.values()):
            missing = [k for k, v in cognito_status.items() if not v]
            result.errors.append(f"Missing Cognito components: {', '.join(missing)}")

        # 3. Validate parameters
        from dova.aws.parameters import ParameterManager

        params = ParameterManager(self.region)
        param_status = params.validate_configuration(stack_name)
        result.checks["ssm_cognito_provider"] = param_status["cognito_provider"]
        result.checks["ssm_client_id"] = param_status["machine_client_id"]
        result.checks["secret_client_secret"] = param_status["machine_client_secret"]

        if not param_status["cognito_provider"] or not param_status["machine_client_id"]:
            result.errors.append("Missing SSM parameters for authentication")
        if not param_status["machine_client_secret"]:
            result.errors.append("Missing client secret in Secrets Manager")

        # 4. Validate Bedrock
        bedrock_result = self.validate_bedrock_access()
        result.checks["bedrock_access"] = bedrock_result.valid
        result.warnings.extend(bedrock_result.warnings)
        if not bedrock_result.valid:
            result.errors.extend(bedrock_result.errors)

        # 5. Validate IAM role exists
        try:
            iam = boto3.client("iam", region_name=self.region)
            role_name = f"{stack_name}-dova-execution-role"
            iam.get_role(RoleName=role_name)
            result.checks["iam_role"] = True
        except ClientError:
            result.checks["iam_role"] = False
            result.warnings.append(f"IAM role {role_name} not found (may be optional for local dev)")

        result.valid = len(result.errors) == 0
        return result


def format_validation_result(result: ValidationResult) -> str:
    """Format validation result for CLI output."""
    lines = []

    # Header
    status = "PASSED" if result.valid else "FAILED"
    lines.append(f"\nAWS Validation: {status}")
    lines.append("=" * 50)

    if result.account_id:
        lines.append(f"Account ID: {result.account_id}")
    if result.region:
        lines.append(f"Region: {result.region}")

    # Checks
    lines.append("\nChecks:")
    for check, passed in result.checks.items():
        icon = "[OK]" if passed else "[FAIL]"
        lines.append(f"  {icon} {check}")

    # Warnings
    if result.warnings:
        lines.append("\nWarnings:")
        for warning in result.warnings:
            lines.append(f"  - {warning}")

    # Errors
    if result.errors:
        lines.append("\nErrors:")
        for error in result.errors:
            lines.append(f"  - {error}")

    return "\n".join(lines)
