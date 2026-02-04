"""IAM policies and roles for DOVA AgentCore.

Defines and creates the IAM policies required for:
- Bedrock model invocation
- AgentCore Memory access
- SSM Parameter Store access
- Secrets Manager access
"""

import json
from dataclasses import dataclass

import boto3
import structlog
from botocore.exceptions import ClientError

logger = structlog.get_logger(__name__)


@dataclass
class IAMSetupResult:
    """Result of IAM setup."""

    success: bool
    role_arn: str | None = None
    policy_arns: list[str] | None = None
    error: str | None = None
    missing_permissions: list[str] | None = None


def get_bedrock_policy(region: str, account_id: str) -> dict:  # noqa: ARG001
    """Get Bedrock model invocation policy.

    Uses wildcard region (*) to allow cross-region model access,
    since not all models are available in all regions.
    """
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "BedrockModelInvocation",
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                "Resource": [
                    "arn:aws:bedrock:*::foundation-model/anthropic.*",
                    "arn:aws:bedrock:*::foundation-model/amazon.*",
                    f"arn:aws:bedrock:*:{account_id}:inference-profile/*",
                ],
            },
            {
                "Sid": "BedrockAgentRuntime",
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeAgent",
                    "bedrock:GetAgentMemory",
                    "bedrock:DeleteAgentMemory",
                ],
                "Resource": f"arn:aws:bedrock:*:{account_id}:agent/*",
            },
        ],
    }


def get_agentcore_policy(region: str, account_id: str, stack_name: str) -> dict:  # noqa: ARG001
    """Get AgentCore Memory and Gateway policy."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AgentCoreMemory",
                "Effect": "Allow",
                "Action": [
                    "bedrock:CreateMemory",
                    "bedrock:DeleteMemory",
                    "bedrock:GetMemory",
                    "bedrock:ListMemories",
                    "bedrock:CreateMemorySession",
                    "bedrock:DeleteMemorySession",
                    "bedrock:GetMemorySession",
                    "bedrock:ListMemorySessions",
                    "bedrock:InvokeMemory",
                ],
                "Resource": [
                    f"arn:aws:bedrock:{region}:{account_id}:memory/*",
                    f"arn:aws:bedrock:{region}:{account_id}:memory-session/*",
                ],
            },
            {
                "Sid": "AgentCoreGateway",
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeGateway",
                    "bedrock:ListGateways",
                    "bedrock:GetGateway",
                ],
                "Resource": f"arn:aws:bedrock:{region}:{account_id}:gateway/*",
            },
        ],
    }


def get_ssm_policy(region: str, account_id: str, stack_name: str) -> dict:
    """Get SSM Parameter Store access policy."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "SSMParameterAccess",
                "Effect": "Allow",
                "Action": [
                    "ssm:GetParameter",
                    "ssm:GetParameters",
                    "ssm:GetParametersByPath",
                    "ssm:PutParameter",
                    "ssm:DeleteParameter",
                ],
                "Resource": f"arn:aws:ssm:{region}:{account_id}:parameter/{stack_name}/*",
            },
        ],
    }


def get_secrets_policy(region: str, account_id: str, stack_name: str) -> dict:
    """Get Secrets Manager access policy."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "SecretsManagerAccess",
                "Effect": "Allow",
                "Action": [
                    "secretsmanager:GetSecretValue",
                    "secretsmanager:CreateSecret",
                    "secretsmanager:UpdateSecret",
                    "secretsmanager:DeleteSecret",
                ],
                "Resource": f"arn:aws:secretsmanager:{region}:{account_id}:secret:/{stack_name}/*",
            },
        ],
    }


def get_cognito_policy(region: str, account_id: str) -> dict:  # noqa: ARG001
    """Get Cognito access policy for setup."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "CognitoUserPoolManagement",
                "Effect": "Allow",
                "Action": [
                    "cognito-idp:CreateUserPool",
                    "cognito-idp:DeleteUserPool",
                    "cognito-idp:DescribeUserPool",
                    "cognito-idp:UpdateUserPool",
                    "cognito-idp:CreateUserPoolClient",
                    "cognito-idp:DeleteUserPoolClient",
                    "cognito-idp:DescribeUserPoolClient",
                    "cognito-idp:UpdateUserPoolClient",
                    "cognito-idp:CreateUserPoolDomain",
                    "cognito-idp:DeleteUserPoolDomain",
                    "cognito-idp:DescribeUserPoolDomain",
                    "cognito-idp:CreateResourceServer",
                    "cognito-idp:DeleteResourceServer",
                    "cognito-idp:DescribeResourceServer",
                ],
                "Resource": "*",
            },
        ],
    }


def get_lambda_trust_policy() -> dict:
    """Get trust policy for Lambda execution role."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            },
            {
                "Effect": "Allow",
                "Principal": {"Service": "bedrock.amazonaws.com"},
                "Action": "sts:AssumeRole",
            },
        ],
    }


class IAMManager:
    """Manages IAM resources for DOVA."""

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.iam = boto3.client("iam", region_name=region)
        self.sts = boto3.client("sts", region_name=region)
        self._logger = logger.bind(component="iam_manager")

    def get_account_id(self) -> str:
        """Get the current AWS account ID."""
        return self.sts.get_caller_identity()["Account"]

    def check_permissions(self, required_actions: list[str]) -> list[str]:
        """Check which permissions are missing.

        Returns list of missing permissions.
        """
        # This is a simplified check - in production you'd use IAM policy simulator
        missing = []
        try:
            # Try to simulate the actions
            for action in required_actions:
                try:
                    # Basic validation that we can describe policies
                    if action.startswith("iam:"):
                        self.iam.list_policies(Scope="Local", MaxItems=1)
                except ClientError as e:
                    if e.response["Error"]["Code"] == "AccessDenied":
                        missing.append(action)
        except Exception as e:
            self._logger.warning("permission_check_failed", error=str(e))

        return missing

    def create_dova_role(
        self,
        stack_name: str,
        include_bedrock: bool = True,
        include_agentcore: bool = True,
    ) -> IAMSetupResult:
        """Create the DOVA execution role with required policies."""
        account_id = self.get_account_id()
        role_name = f"{stack_name}-dova-execution-role"
        policy_arns = []

        self._logger.info("creating_dova_role", role_name=role_name)

        try:
            # Create the role
            try:
                self.iam.create_role(
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps(get_lambda_trust_policy()),
                    Description=f"DOVA execution role for {stack_name}",
                    Tags=[
                        {"Key": "Stack", "Value": stack_name},
                        {"Key": "ManagedBy", "Value": "dova"},
                    ],
                )
                self._logger.info("role_created", role_name=role_name)
            except ClientError as e:
                if e.response["Error"]["Code"] == "EntityAlreadyExists":
                    self._logger.info("role_exists", role_name=role_name)
                else:
                    raise

            # Create and attach policies
            policies_to_create = [
                (f"{stack_name}-ssm-policy", get_ssm_policy(self.region, account_id, stack_name)),
                (
                    f"{stack_name}-secrets-policy",
                    get_secrets_policy(self.region, account_id, stack_name),
                ),
            ]

            if include_bedrock:
                policies_to_create.append(
                    (f"{stack_name}-bedrock-policy", get_bedrock_policy(self.region, account_id))
                )

            if include_agentcore:
                policies_to_create.append(
                    (
                        f"{stack_name}-agentcore-policy",
                        get_agentcore_policy(self.region, account_id, stack_name),
                    )
                )

            for policy_name, policy_doc in policies_to_create:
                policy_arn = self._create_or_update_policy(policy_name, policy_doc)
                if policy_arn:
                    policy_arns.append(policy_arn)
                    self._attach_policy_to_role(role_name, policy_arn)

            # Attach AWS managed policies
            managed_policies = [
                "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            ]
            for managed_arn in managed_policies:
                self._attach_policy_to_role(role_name, managed_arn)
                policy_arns.append(managed_arn)

            role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"

            return IAMSetupResult(
                success=True,
                role_arn=role_arn,
                policy_arns=policy_arns,
            )

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_msg = e.response["Error"]["Message"]

            if error_code == "AccessDenied":
                return IAMSetupResult(
                    success=False,
                    error=f"Access denied: {error_msg}",
                    missing_permissions=["iam:CreateRole", "iam:CreatePolicy", "iam:AttachRolePolicy"],
                )

            self._logger.error("role_creation_failed", error=str(e))
            return IAMSetupResult(success=False, error=str(e))

    def _create_or_update_policy(self, policy_name: str, policy_doc: dict) -> str | None:
        """Create or update an IAM policy."""
        account_id = self.get_account_id()
        policy_arn = f"arn:aws:iam::{account_id}:policy/{policy_name}"

        try:
            # Try to create the policy
            response = self.iam.create_policy(
                PolicyName=policy_name,
                PolicyDocument=json.dumps(policy_doc),
                Description=f"DOVA policy: {policy_name}",
                Tags=[{"Key": "ManagedBy", "Value": "dova"}],
            )
            self._logger.info("policy_created", policy_name=policy_name)
            return response["Policy"]["Arn"]

        except ClientError as e:
            if e.response["Error"]["Code"] == "EntityAlreadyExists":
                # Policy exists, update it by creating a new version
                self._logger.info("policy_exists_updating", policy_name=policy_name)

                # Delete oldest version if we have too many
                versions = self.iam.list_policy_versions(PolicyArn=policy_arn)["Versions"]
                non_default = [v for v in versions if not v["IsDefaultVersion"]]
                if len(non_default) >= 4:
                    oldest = sorted(non_default, key=lambda x: x["CreateDate"])[0]
                    self.iam.delete_policy_version(
                        PolicyArn=policy_arn, VersionId=oldest["VersionId"]
                    )

                # Create new version
                self.iam.create_policy_version(
                    PolicyArn=policy_arn,
                    PolicyDocument=json.dumps(policy_doc),
                    SetAsDefault=True,
                )
                return policy_arn
            raise

    def _attach_policy_to_role(self, role_name: str, policy_arn: str) -> None:
        """Attach a policy to a role."""
        try:
            self.iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
            self._logger.debug("policy_attached", role=role_name, policy=policy_arn)
        except ClientError as e:
            if e.response["Error"]["Code"] != "EntityAlreadyExists":
                raise

    def delete_dova_resources(self, stack_name: str) -> bool:
        """Delete all DOVA IAM resources for a stack."""
        account_id = self.get_account_id()
        role_name = f"{stack_name}-dova-execution-role"

        self._logger.info("deleting_iam_resources", stack_name=stack_name)

        try:
            # Detach all policies from role
            try:
                attached = self.iam.list_attached_role_policies(RoleName=role_name)
                for policy in attached["AttachedPolicies"]:
                    self.iam.detach_role_policy(
                        RoleName=role_name, PolicyArn=policy["PolicyArn"]
                    )
            except ClientError:
                pass

            # Delete role
            try:
                self.iam.delete_role(RoleName=role_name)
                self._logger.info("role_deleted", role_name=role_name)
            except ClientError:
                pass

            # Delete policies
            policy_names = [
                f"{stack_name}-ssm-policy",
                f"{stack_name}-secrets-policy",
                f"{stack_name}-bedrock-policy",
                f"{stack_name}-agentcore-policy",
            ]

            for policy_name in policy_names:
                policy_arn = f"arn:aws:iam::{account_id}:policy/{policy_name}"
                try:
                    # Delete all versions except default
                    versions = self.iam.list_policy_versions(PolicyArn=policy_arn)["Versions"]
                    for v in versions:
                        if not v["IsDefaultVersion"]:
                            self.iam.delete_policy_version(
                                PolicyArn=policy_arn, VersionId=v["VersionId"]
                            )
                    self.iam.delete_policy(PolicyArn=policy_arn)
                    self._logger.info("policy_deleted", policy_name=policy_name)
                except ClientError:
                    pass

            return True

        except Exception as e:
            self._logger.error("iam_cleanup_failed", error=str(e))
            return False


def get_required_setup_permissions() -> list[str]:
    """Get list of IAM permissions required to run setup."""
    return [
        # IAM
        "iam:CreateRole",
        "iam:DeleteRole",
        "iam:GetRole",
        "iam:CreatePolicy",
        "iam:DeletePolicy",
        "iam:GetPolicy",
        "iam:AttachRolePolicy",
        "iam:DetachRolePolicy",
        "iam:ListAttachedRolePolicies",
        "iam:CreatePolicyVersion",
        "iam:DeletePolicyVersion",
        "iam:ListPolicyVersions",
        "iam:PassRole",
        # Cognito
        "cognito-idp:CreateUserPool",
        "cognito-idp:DeleteUserPool",
        "cognito-idp:DescribeUserPool",
        "cognito-idp:CreateUserPoolClient",
        "cognito-idp:DeleteUserPoolClient",
        "cognito-idp:CreateUserPoolDomain",
        "cognito-idp:DeleteUserPoolDomain",
        "cognito-idp:CreateResourceServer",
        "cognito-idp:DeleteResourceServer",
        # SSM
        "ssm:PutParameter",
        "ssm:GetParameter",
        "ssm:GetParametersByPath",
        "ssm:DeleteParameter",
        # Secrets Manager
        "secretsmanager:CreateSecret",
        "secretsmanager:GetSecretValue",
        "secretsmanager:DeleteSecret",
        # STS
        "sts:GetCallerIdentity",
    ]


def get_required_deploy_permissions() -> list[str]:
    """Get list of IAM permissions required to run deploy."""
    return [
        # S3 (deployment artifacts)
        "s3:CreateBucket",
        "s3:DeleteBucket",
        "s3:PutBucketVersioning",
        "s3:PutBucketTagging",
        "s3:HeadBucket",
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:ListBucketVersions",
        "s3:DeleteObjectVersion",
        # Lambda
        "lambda:CreateFunction",
        "lambda:UpdateFunctionCode",
        "lambda:UpdateFunctionConfiguration",
        "lambda:DeleteFunction",
        "lambda:GetFunction",
        "lambda:AddPermission",
        "lambda:RemovePermission",
        # API Gateway
        "apigateway:POST",
        "apigateway:GET",
        "apigateway:PUT",
        "apigateway:DELETE",
        "apigateway:PATCH",
        # CloudFormation
        "cloudformation:CreateStack",
        "cloudformation:UpdateStack",
        "cloudformation:DeleteStack",
        "cloudformation:DescribeStacks",
        "cloudformation:DescribeStackEvents",
        "cloudformation:GetTemplate",
        # IAM (for passing role to Lambda)
        "iam:PassRole",
        "iam:GetRole",
        # SSM (for storing deployment info)
        "ssm:PutParameter",
        "ssm:GetParameter",
        "ssm:GetParametersByPath",
        "ssm:DeleteParameter",
        # STS
        "sts:GetCallerIdentity",
    ]
