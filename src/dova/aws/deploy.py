"""Deployment Manager for DOVA Lambda deployment.

Orchestrates the full deployment workflow:
1. Package Lambda code
2. Upload to S3
3. Generate CloudFormation template
4. Create/update CloudFormation stack
5. Wait for completion
6. Store deployment info in SSM
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum

import boto3
import structlog
from botocore.exceptions import ClientError

from dova.aws.cloudformation import CloudFormationBuilder, CloudFormationConfig
from dova.aws.lambda_packager import LambdaPackager
from dova.aws.parameters import ParameterManager
from dova.aws.s3_manager import S3Manager

logger = structlog.get_logger(__name__)


class DeployPhase(Enum):
    """Phases of deployment."""

    PACKAGE = "package"
    UPLOAD = "upload"
    TEMPLATE = "template"
    DEPLOY = "deploy"
    WAIT = "wait"
    FINALIZE = "finalize"


@dataclass
class DeployConfig:
    """Configuration for Lambda deployment."""

    stack_name: str
    region: str = "us-east-1"
    lambda_memory: int = 1024
    lambda_timeout: int = 300
    enable_cognito: bool = False
    enable_cors: bool = True


@dataclass
class DeployResult:
    """Result of deployment operation."""

    success: bool
    phase: DeployPhase | None = None
    lambda_arn: str | None = None
    api_url: str | None = None
    stack_id: str | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class DeployStatus:
    """Status of a deployment."""

    stack_name: str
    status: str
    lambda_arn: str | None = None
    api_url: str | None = None
    last_updated: str | None = None
    outputs: dict = field(default_factory=dict)


class DeployManager:
    """Orchestrates DOVA Lambda deployment."""

    def __init__(self, config: DeployConfig):
        self.config = config
        self.region = config.region
        self.cfn = boto3.client("cloudformation", region_name=config.region)
        self.s3_manager = S3Manager(config.region)
        self.packager = LambdaPackager()
        self.cf_builder = CloudFormationBuilder()
        self.param_manager = ParameterManager(config.region)
        self._logger = logger.bind(
            component="deploy_manager",
            stack_name=config.stack_name,
        )

    def deploy(self) -> DeployResult:
        """Run the full deployment workflow.

        Returns:
            DeployResult with deployment details
        """
        result = DeployResult(success=False)

        self._logger.info("starting_deployment", config=self.config)

        try:
            # Phase 1: Package Lambda code
            result.phase = DeployPhase.PACKAGE
            self._logger.info("phase_package")

            package_result = self.packager.create_package()
            if not package_result.success:
                result.errors.append(f"Packaging failed: {package_result.error}")
                return result

            # Phase 2: Upload to S3
            result.phase = DeployPhase.UPLOAD
            self._logger.info("phase_upload")

            bucket = self.s3_manager.ensure_deployment_bucket(self.config.stack_name)
            if not bucket:
                result.errors.append("Failed to create deployment bucket")
                return result

            upload_result = self.s3_manager.upload_lambda_package(
                bucket,
                package_result.zip_path,  # type: ignore
                self.config.stack_name,
            )
            if not upload_result.success:
                result.errors.append(f"Upload failed: {upload_result.error}")
                return result

            # Phase 3: Generate CloudFormation template
            result.phase = DeployPhase.TEMPLATE
            self._logger.info("phase_template")

            # Get role ARN from existing setup
            role_arn = self._get_role_arn()
            if not role_arn:
                result.errors.append(
                    f"IAM role not found for stack {self.config.stack_name}. "
                    "Run 'dova aws setup' first."
                )
                return result

            # Get Cognito user pool ARN if enabled
            cognito_arn = None
            if self.config.enable_cognito:
                cognito_arn = self._get_cognito_user_pool_arn()

            cf_config = CloudFormationConfig(
                stack_name=self.config.stack_name,
                lambda_s3_bucket=upload_result.bucket,  # type: ignore
                lambda_s3_key=upload_result.key,  # type: ignore
                role_arn=role_arn,
                region=self.config.region,
                lambda_memory=self.config.lambda_memory,
                lambda_timeout=self.config.lambda_timeout,
                cognito_user_pool_arn=cognito_arn,
                enable_cors=self.config.enable_cors,
            )

            template = self.cf_builder.build_template(cf_config)

            # Phase 4: Deploy CloudFormation stack
            result.phase = DeployPhase.DEPLOY
            self._logger.info("phase_deploy")

            stack_id = self._deploy_stack(template)
            if not stack_id:
                result.errors.append("Failed to create/update CloudFormation stack")
                return result

            result.stack_id = stack_id

            # Phase 5: Wait for completion
            result.phase = DeployPhase.WAIT
            self._logger.info("phase_wait")

            if not self._wait_for_stack():
                result.errors.append("Stack deployment failed or timed out")
                return result

            # Phase 6: Finalize
            result.phase = DeployPhase.FINALIZE
            self._logger.info("phase_finalize")

            # Get outputs
            outputs = self._get_stack_outputs()
            result.lambda_arn = outputs.get("LambdaFunctionArn")
            result.api_url = outputs.get("ApiGatewayUrl")

            # Store deployment info in SSM
            self._store_deployment_info(outputs)

            result.success = True
            self._logger.info(
                "deployment_complete",
                lambda_arn=result.lambda_arn,
                api_url=result.api_url,
            )

            return result

        except Exception as e:
            self._logger.exception("deployment_failed", error=str(e))
            result.errors.append(f"Unexpected error: {str(e)}")
            return result

    def get_status(self) -> DeployStatus | None:
        """Get the current deployment status.

        Returns:
            DeployStatus if stack exists, None otherwise
        """
        stack_name = f"{self.config.stack_name}-dova-deploy"

        try:
            response = self.cfn.describe_stacks(StackName=stack_name)
            stack = response["Stacks"][0]

            outputs = {}
            for output in stack.get("Outputs", []):
                outputs[output["OutputKey"]] = output["OutputValue"]

            return DeployStatus(
                stack_name=stack_name,
                status=stack["StackStatus"],
                lambda_arn=outputs.get("LambdaFunctionArn"),
                api_url=outputs.get("ApiGatewayUrl"),
                last_updated=stack.get("LastUpdatedTime", stack.get("CreationTime")),
                outputs=outputs,
            )

        except ClientError as e:
            if "does not exist" in str(e):
                return None
            raise

    def delete_deployment(self) -> bool:
        """Delete the deployment stack and artifacts.

        Returns:
            True if successful
        """
        stack_name = f"{self.config.stack_name}-dova-deploy"

        self._logger.info("deleting_deployment", stack=stack_name)

        try:
            # Delete CloudFormation stack
            try:
                self.cfn.delete_stack(StackName=stack_name)

                # Wait for deletion
                waiter = self.cfn.get_waiter("stack_delete_complete")
                waiter.wait(StackName=stack_name, WaiterConfig={"MaxAttempts": 30})
            except ClientError as e:
                if "does not exist" not in str(e):
                    raise

            # Delete S3 artifacts
            self.s3_manager.delete_deployment_artifacts(self.config.stack_name)

            # Delete deployment SSM parameters
            self._delete_deployment_info()

            self._logger.info("deployment_deleted")
            return True

        except Exception as e:
            self._logger.error("deletion_failed", error=str(e))
            return False

    def _get_role_arn(self) -> str | None:
        """Get the IAM role ARN from SSM or construct it."""
        # Try to get from SSM first
        role_arn = self.param_manager.get_parameter(
            f"/{self.config.stack_name}/role_arn"
        )
        if role_arn:
            return role_arn

        # Construct the expected role ARN
        try:
            sts = boto3.client("sts", region_name=self.region)
            account_id = sts.get_caller_identity()["Account"]
            role_name = f"{self.config.stack_name}-dova-execution-role"

            # Verify role exists
            iam = boto3.client("iam", region_name=self.region)
            iam.get_role(RoleName=role_name)

            return f"arn:aws:iam::{account_id}:role/{role_name}"

        except ClientError:
            return None

    def _get_cognito_user_pool_arn(self) -> str | None:
        """Get the Cognito user pool ARN."""
        # Try to get from SSM
        user_pool_id = self.param_manager.get_parameter(
            f"/{self.config.stack_name}/cognito_user_pool_id"
        )
        if not user_pool_id:
            return None

        try:
            sts = boto3.client("sts", region_name=self.region)
            account_id = sts.get_caller_identity()["Account"]
            return f"arn:aws:cognito-idp:{self.region}:{account_id}:userpool/{user_pool_id}"
        except ClientError:
            return None

    def _deploy_stack(self, template: dict) -> str | None:
        """Create or update the CloudFormation stack.

        Returns:
            Stack ID if successful
        """
        stack_name = f"{self.config.stack_name}-dova-deploy"

        try:
            # Check if stack exists
            try:
                self.cfn.describe_stacks(StackName=stack_name)
                stack_exists = True
            except ClientError:
                stack_exists = False

            template_body = json.dumps(template)

            if stack_exists:
                self._logger.info("updating_stack", stack=stack_name)
                try:
                    response = self.cfn.update_stack(
                        StackName=stack_name,
                        TemplateBody=template_body,
                        Capabilities=["CAPABILITY_IAM"],
                        Tags=[
                            {"Key": "ManagedBy", "Value": "dova"},
                            {"Key": "Stack", "Value": self.config.stack_name},
                        ],
                    )
                    return response["StackId"]
                except ClientError as e:
                    if "No updates are to be performed" in str(e):
                        self._logger.info("no_updates_needed")
                        response = self.cfn.describe_stacks(StackName=stack_name)
                        return response["Stacks"][0]["StackId"]
                    raise
            else:
                self._logger.info("creating_stack", stack=stack_name)
                response = self.cfn.create_stack(
                    StackName=stack_name,
                    TemplateBody=template_body,
                    Capabilities=["CAPABILITY_IAM"],
                    Tags=[
                        {"Key": "ManagedBy", "Value": "dova"},
                        {"Key": "Stack", "Value": self.config.stack_name},
                    ],
                    OnFailure="DELETE",
                )
                return response["StackId"]

        except ClientError as e:
            self._logger.error("stack_operation_failed", error=str(e))
            return None

    def _wait_for_stack(self, timeout: int = 600) -> bool:
        """Wait for stack operation to complete.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            True if stack reached a successful state
        """
        stack_name = f"{self.config.stack_name}-dova-deploy"
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = self.cfn.describe_stacks(StackName=stack_name)
                status = response["Stacks"][0]["StackStatus"]

                self._logger.debug("stack_status", status=status)

                if status in ("CREATE_COMPLETE", "UPDATE_COMPLETE"):
                    return True
                elif status in (
                    "CREATE_FAILED",
                    "ROLLBACK_COMPLETE",
                    "UPDATE_ROLLBACK_COMPLETE",
                    "DELETE_COMPLETE",
                ):
                    self._logger.error("stack_failed", status=status)
                    return False
                elif "_IN_PROGRESS" in status:
                    time.sleep(10)
                else:
                    self._logger.warning("unexpected_stack_status", status=status)
                    time.sleep(10)

            except ClientError as e:
                if "does not exist" in str(e):
                    self._logger.error("stack_deleted_unexpectedly")
                    return False
                raise

        self._logger.error("stack_timeout", timeout=timeout)
        return False

    def _get_stack_outputs(self) -> dict:
        """Get CloudFormation stack outputs."""
        stack_name = f"{self.config.stack_name}-dova-deploy"

        try:
            response = self.cfn.describe_stacks(StackName=stack_name)
            outputs = {}
            for output in response["Stacks"][0].get("Outputs", []):
                outputs[output["OutputKey"]] = output["OutputValue"]
            return outputs
        except ClientError:
            return {}

    def _store_deployment_info(self, outputs: dict) -> None:
        """Store deployment information in SSM."""
        for key, value in outputs.items():
            param_name = f"/{self.config.stack_name}/deploy/{key}"
            self.param_manager._put_parameter(param_name, value)

    def _delete_deployment_info(self) -> None:
        """Delete deployment SSM parameters."""
        prefix = f"/{self.config.stack_name}/deploy/"

        try:
            response = self.param_manager.ssm.get_parameters_by_path(
                Path=prefix,
                Recursive=True,
            )

            for param in response.get("Parameters", []):
                self.param_manager.ssm.delete_parameter(Name=param["Name"])

        except ClientError:
            pass


def format_deploy_result(result: DeployResult) -> str:
    """Format deployment result for CLI output."""
    lines = []

    status = "SUCCESS" if result.success else "FAILED"
    lines.append(f"\nDeployment: {status}")
    lines.append("=" * 50)

    if result.phase:
        lines.append(f"Phase: {result.phase.value}")

    if result.lambda_arn:
        lines.append(f"\nLambda ARN: {result.lambda_arn}")

    if result.api_url:
        lines.append(f"API URL: {result.api_url}")

    if result.stack_id:
        lines.append(f"Stack ID: {result.stack_id}")

    if result.errors:
        lines.append("\nErrors:")
        for error in result.errors:
            lines.append(f"  - {error}")

    if result.success:
        lines.append("\nNext Steps:")
        lines.append("  1. Test the API endpoint:")
        lines.append(f"     curl -X POST {result.api_url} \\")
        lines.append('       -H "Content-Type: application/json" \\')
        lines.append('       -d \'{"prompt": "What is BERT?"}\'')

    return "\n".join(lines)


def format_deploy_status(status: DeployStatus | None) -> str:
    """Format deployment status for CLI output."""
    if not status:
        return "No deployment found."

    lines = []
    lines.append(f"\nDeployment Status: {status.status}")
    lines.append("=" * 50)
    lines.append(f"Stack Name: {status.stack_name}")

    if status.lambda_arn:
        lines.append(f"Lambda ARN: {status.lambda_arn}")

    if status.api_url:
        lines.append(f"API URL: {status.api_url}")

    if status.last_updated:
        lines.append(f"Last Updated: {status.last_updated}")

    return "\n".join(lines)
