"""DOVA AWS Setup Module.

Automated setup for AWS services required by DOVA AgentCore integration:
- Cognito (OAuth2 authentication)
- IAM (policies and roles)
- SSM Parameter Store (configuration)
- Secrets Manager (credentials)
- Bedrock (model access validation)
"""

from dova.aws.setup import AWSSetup, SetupConfig, SetupResult

__all__ = ["AWSSetup", "SetupConfig", "SetupResult"]
