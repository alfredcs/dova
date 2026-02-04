"""Main AWS setup orchestration for DOVA.

Provides automated setup of all AWS services required for DOVA AgentCore:
- Cognito (OAuth2 authentication)
- IAM (policies and roles)
- SSM Parameter Store (configuration)
- Secrets Manager (credentials)
- Bedrock (model access validation)
"""

from dataclasses import dataclass, field
from enum import Enum

import structlog

from dova.aws.cognito import CognitoManager, CognitoSetupResult
from dova.aws.iam import IAMManager, IAMSetupResult, get_required_setup_permissions
from dova.aws.parameters import ParameterManager, ParameterSetupResult
from dova.aws.validators import AWSValidator, ValidationResult

logger = structlog.get_logger(__name__)


class SetupPhase(Enum):
    """Phases of AWS setup."""

    VALIDATE_CREDENTIALS = "validate_credentials"
    VALIDATE_PERMISSIONS = "validate_permissions"
    CREATE_IAM = "create_iam"
    CREATE_COGNITO = "create_cognito"
    STORE_PARAMETERS = "store_parameters"
    VALIDATE_BEDROCK = "validate_bedrock"
    FINALIZE = "finalize"


@dataclass
class SetupConfig:
    """Configuration for AWS setup."""

    stack_name: str
    region: str = "us-east-1"
    include_bedrock: bool = True
    include_agentcore: bool = True
    gateway_url: str | None = None
    memory_id: str | None = None
    generate_env_file: bool = True
    env_file_path: str = ".env.aws"


@dataclass
class SetupResult:
    """Result of AWS setup."""

    success: bool
    phase: SetupPhase | None = None
    iam_result: IAMSetupResult | None = None
    cognito_result: CognitoSetupResult | None = None
    parameter_result: ParameterSetupResult | None = None
    validation_result: ValidationResult | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    env_file_path: str | None = None


class AWSSetup:
    """Orchestrates AWS setup for DOVA.

    Usage:
        config = SetupConfig(stack_name="my-dova-stack", region="us-east-1")
        setup = AWSSetup(config)
        result = setup.run()

        if result.success:
            print("Setup complete!")
        else:
            print(f"Setup failed at phase {result.phase}: {result.errors}")
    """

    def __init__(self, config: SetupConfig):
        self.config = config
        self.validator = AWSValidator(config.region)
        self.iam_manager = IAMManager(config.region)
        self.cognito_manager = CognitoManager(config.region)
        self.parameter_manager = ParameterManager(config.region)
        self._logger = logger.bind(
            component="aws_setup",
            stack_name=config.stack_name,
        )

    def run(self, skip_validation: bool = False) -> SetupResult:
        """Run the complete AWS setup.

        Args:
            skip_validation: Skip pre-flight validation (not recommended)

        Returns:
            SetupResult with all component results
        """
        result = SetupResult(success=False)

        self._logger.info("starting_aws_setup", config=self.config)

        try:
            # Phase 1: Validate credentials
            result.phase = SetupPhase.VALIDATE_CREDENTIALS
            if not skip_validation:
                cred_result = self.validator.validate_credentials()
                if not cred_result.valid:
                    result.errors.extend(cred_result.errors)
                    return result
                self._logger.info("credentials_validated", account=cred_result.account_id)

            # Phase 2: Validate permissions
            result.phase = SetupPhase.VALIDATE_PERMISSIONS
            if not skip_validation:
                perm_result = self.validator.validate_iam_permissions(
                    get_required_setup_permissions()
                )
                if not perm_result.valid:
                    result.errors.append(
                        "Missing required IAM permissions. Please ensure your AWS user/role has:\n"
                        + "\n".join(f"  - {err}" for err in perm_result.errors)
                    )
                    result.errors.append(
                        "\nYou can add these permissions by attaching a policy with the following actions:\n"
                        + self._format_missing_permissions_policy(perm_result)
                    )
                    return result
                self._logger.info("permissions_validated")

            # Phase 3: Create IAM resources
            result.phase = SetupPhase.CREATE_IAM
            iam_result = self.iam_manager.create_dova_role(
                self.config.stack_name,
                include_bedrock=self.config.include_bedrock,
                include_agentcore=self.config.include_agentcore,
            )
            result.iam_result = iam_result
            if not iam_result.success:
                result.errors.append(f"IAM setup failed: {iam_result.error}")
                if iam_result.missing_permissions:
                    result.errors.append(
                        "Missing permissions: " + ", ".join(iam_result.missing_permissions)
                    )
                return result
            self._logger.info("iam_created", role_arn=iam_result.role_arn)

            # Phase 4: Create Cognito resources
            result.phase = SetupPhase.CREATE_COGNITO
            cognito_result = self.cognito_manager.setup_cognito(self.config.stack_name)
            result.cognito_result = cognito_result
            if not cognito_result.success:
                result.errors.append(f"Cognito setup failed: {cognito_result.error}")
                return result
            self._logger.info(
                "cognito_created",
                user_pool=cognito_result.user_pool_id,
                client_id=cognito_result.client_id,
            )

            # Phase 5: Store parameters
            result.phase = SetupPhase.STORE_PARAMETERS
            param_result = self.parameter_manager.store_configuration(
                stack_name=self.config.stack_name,
                cognito_domain=cognito_result.domain or "",
                client_id=cognito_result.client_id or "",
                client_secret=cognito_result.client_secret or "",
                gateway_url=self.config.gateway_url,
                memory_id=self.config.memory_id,
            )
            result.parameter_result = param_result
            if not param_result.success:
                result.errors.append(f"Parameter storage failed: {param_result.error}")
                return result
            self._logger.info("parameters_stored")

            # Phase 6: Validate Bedrock
            result.phase = SetupPhase.VALIDATE_BEDROCK
            if self.config.include_bedrock:
                bedrock_result = self.validator.validate_bedrock_access()
                if not bedrock_result.valid:
                    result.warnings.extend(bedrock_result.errors)
                    self._logger.warning(
                        "bedrock_validation_warnings",
                        warnings=bedrock_result.errors,
                    )
                else:
                    self._logger.info("bedrock_validated")

            # Phase 7: Finalize
            result.phase = SetupPhase.FINALIZE
            if self.config.generate_env_file:
                self.parameter_manager.generate_env_file(
                    self.config.stack_name,
                    self.config.env_file_path,
                )
                result.env_file_path = self.config.env_file_path

            # Run final validation
            result.validation_result = self.validator.validate_complete_setup(
                self.config.stack_name
            )

            result.success = True
            self._logger.info("aws_setup_complete")

            return result

        except Exception as e:
            self._logger.exception("aws_setup_failed", error=str(e))
            result.errors.append(f"Unexpected error: {str(e)}")
            return result

    def teardown(self) -> SetupResult:
        """Remove all AWS resources created by setup.

        WARNING: This will delete all DOVA resources for the stack,
        including any Lambda deployments.

        Returns:
            SetupResult with teardown status
        """
        result = SetupResult(success=False)

        self._logger.info("starting_teardown", stack_name=self.config.stack_name)

        try:
            # Delete in reverse order of creation

            # 0. Delete deployment resources first (if any)
            try:
                from dova.aws.deploy import DeployConfig, DeployManager

                deploy_config = DeployConfig(
                    stack_name=self.config.stack_name,
                    region=self.config.region,
                )
                deploy_manager = DeployManager(deploy_config)
                deploy_manager.delete_deployment()
                self._logger.info("deployment_deleted")
            except Exception as e:
                self._logger.warning("deployment_deletion_skipped", error=str(e))

            # 1. Delete parameters
            self.parameter_manager.delete_configuration(self.config.stack_name)

            # 2. Delete Cognito
            self.cognito_manager.delete_cognito_resources(self.config.stack_name)

            # 3. Delete IAM
            self.iam_manager.delete_dova_resources(self.config.stack_name)

            result.success = True
            self._logger.info("teardown_complete")

        except Exception as e:
            self._logger.exception("teardown_failed", error=str(e))
            result.errors.append(f"Teardown error: {str(e)}")

        return result

    def validate(self) -> ValidationResult:
        """Validate current AWS setup.

        Returns:
            ValidationResult with all checks
        """
        return self.validator.validate_complete_setup(self.config.stack_name)

    def _format_missing_permissions_policy(self, result: ValidationResult) -> str:
        """Format missing permissions as an IAM policy for easy copy-paste."""
        missing_actions = [
            check for check, passed in result.checks.items() if not passed
        ]

        if not missing_actions:
            return ""

        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": missing_actions,
                    "Resource": "*",
                }
            ],
        }

        import json

        return json.dumps(policy, indent=2)


def run_interactive_setup(region: str = "us-east-1") -> SetupResult:
    """Run interactive setup with prompts.

    This is called by the CLI for guided setup.

    Args:
        region: AWS region to use

    Returns:
        SetupResult from the setup process
    """
    import os

    # Get stack name
    stack_name = os.environ.get("STACK_NAME")
    if not stack_name:
        # Generate a default
        import secrets as sec

        stack_name = f"dova-{sec.token_hex(4)}"

    config = SetupConfig(
        stack_name=stack_name,
        region=region,
    )

    setup = AWSSetup(config)
    return setup.run()


def format_setup_result(result: SetupResult) -> str:
    """Format setup result for CLI output."""
    lines = []

    # Header
    status = "SUCCESS" if result.success else "FAILED"
    lines.append(f"\nAWS Setup: {status}")
    lines.append("=" * 50)

    if result.phase:
        lines.append(f"Completed phase: {result.phase.value}")

    # IAM
    if result.iam_result:
        lines.append("\nIAM Resources:")
        if result.iam_result.role_arn:
            lines.append(f"  Role ARN: {result.iam_result.role_arn}")
        if result.iam_result.policy_arns:
            lines.append(f"  Policies: {len(result.iam_result.policy_arns)} created")

    # Cognito
    if result.cognito_result:
        lines.append("\nCognito Resources:")
        if result.cognito_result.user_pool_id:
            lines.append(f"  User Pool ID: {result.cognito_result.user_pool_id}")
        if result.cognito_result.client_id:
            lines.append(f"  Client ID: {result.cognito_result.client_id}")
        if result.cognito_result.domain:
            lines.append(f"  Domain: {result.cognito_result.domain}")

    # Parameters
    if result.parameter_result:
        lines.append("\nStored Configuration:")
        if result.parameter_result.parameters_created:
            for param in result.parameter_result.parameters_created:
                lines.append(f"  SSM: {param}")
        if result.parameter_result.secrets_created:
            for secret in result.parameter_result.secrets_created:
                lines.append(f"  Secret: {secret}")

    # Env file
    if result.env_file_path:
        lines.append(f"\nEnvironment file: {result.env_file_path}")

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

    # Next steps
    if result.success:
        lines.append("\nNext Steps:")
        lines.append("  1. Source the environment file:")
        lines.append(f"     source {result.env_file_path or '.env.aws'}")
        lines.append("  2. Start DOVA in AgentCore mode:")
        lines.append("     dova serve --mode agentcore")

    return "\n".join(lines)
