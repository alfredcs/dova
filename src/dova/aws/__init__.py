"""DOVA AWS Setup and Deployment Module.

Automated setup for AWS services required by DOVA AgentCore integration:
- Cognito (OAuth2 authentication)
- IAM (policies and roles)
- SSM Parameter Store (configuration)
- Secrets Manager (credentials)
- Bedrock (model access validation)
- Lambda deployment
- API Gateway integration
"""

from dova.aws.cloudformation import CloudFormationBuilder, CloudFormationConfig
from dova.aws.deploy import DeployConfig, DeployManager, DeployResult
from dova.aws.lambda_packager import LambdaPackager, PackageResult
from dova.aws.s3_manager import S3Manager, S3UploadResult
from dova.aws.setup import AWSSetup, SetupConfig, SetupResult

__all__ = [
    # Setup
    "AWSSetup",
    "SetupConfig",
    "SetupResult",
    # Deploy
    "DeployManager",
    "DeployConfig",
    "DeployResult",
    # CloudFormation
    "CloudFormationBuilder",
    "CloudFormationConfig",
    # Lambda Packaging
    "LambdaPackager",
    "PackageResult",
    # S3
    "S3Manager",
    "S3UploadResult",
]
