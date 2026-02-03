"""AWS Configuration utilities (SSM Parameters, Secrets Manager).

Provides functions to fetch configuration from AWS SSM Parameter Store
and AWS Secrets Manager for AgentCore integration.
"""

import os
from functools import lru_cache

import boto3
import structlog

logger = structlog.get_logger(__name__)


def get_aws_region() -> str:
    """Get AWS region from environment or default."""
    return os.environ.get("AWS_REGION", "us-east-1")


@lru_cache(maxsize=100)
def get_ssm_parameter(parameter_name: str, with_decryption: bool = True) -> str:
    """Fetch parameter from SSM Parameter Store.

    Args:
        parameter_name: Full parameter name (e.g., "/mystack/cognito_provider")
        with_decryption: Whether to decrypt SecureString parameters

    Returns:
        Parameter value

    Raises:
        botocore.exceptions.ClientError: If parameter not found or access denied
    """
    region = get_aws_region()
    ssm = boto3.client("ssm", region_name=region)

    logger.debug("ssm_get_parameter", parameter=parameter_name, region=region)

    response = ssm.get_parameter(Name=parameter_name, WithDecryption=with_decryption)
    return response["Parameter"]["Value"]


def get_ssm_parameters_by_path(
    path: str,
    recursive: bool = True,
    with_decryption: bool = True,
) -> dict[str, str]:
    """Fetch all parameters under a path from SSM Parameter Store.

    Args:
        path: Parameter path prefix (e.g., "/mystack/")
        recursive: Whether to fetch parameters recursively
        with_decryption: Whether to decrypt SecureString parameters

    Returns:
        Dictionary mapping parameter names to values
    """
    region = get_aws_region()
    ssm = boto3.client("ssm", region_name=region)

    logger.debug("ssm_get_parameters_by_path", path=path, region=region)

    parameters = {}
    paginator = ssm.get_paginator("get_parameters_by_path")

    for page in paginator.paginate(
        Path=path,
        Recursive=recursive,
        WithDecryption=with_decryption,
    ):
        for param in page.get("Parameters", []):
            # Extract name without path prefix
            name = param["Name"]
            parameters[name] = param["Value"]

    return parameters


def get_secret(secret_name: str) -> str:
    """Fetch secret from AWS Secrets Manager.

    Args:
        secret_name: Secret name or ARN

    Returns:
        Secret string value

    Raises:
        botocore.exceptions.ClientError: If secret not found or access denied
    """
    region = get_aws_region()
    client = boto3.client("secretsmanager", region_name=region)

    logger.debug("secretsmanager_get_secret", secret=secret_name, region=region)

    response = client.get_secret_value(SecretId=secret_name)
    return response["SecretString"]


def get_secret_json(secret_name: str) -> dict:
    """Fetch and parse JSON secret from AWS Secrets Manager.

    Args:
        secret_name: Secret name or ARN

    Returns:
        Parsed JSON as dictionary

    Raises:
        botocore.exceptions.ClientError: If secret not found or access denied
        json.JSONDecodeError: If secret is not valid JSON
    """
    import json

    secret_string = get_secret(secret_name)
    return json.loads(secret_string)


def clear_ssm_cache() -> None:
    """Clear the SSM parameter cache."""
    get_ssm_parameter.cache_clear()
