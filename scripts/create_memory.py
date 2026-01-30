#!/usr/bin/env python3
"""
DOVA Memory Setup Script.

This script sets up the AgentCore Memory for DOVA,
creating the required memory namespaces and configuring
memory strategies for user profiling.
"""

import asyncio
import json
import os
import sys
from datetime import datetime

import boto3
from botocore.exceptions import ClientError


# Memory configuration
MEMORY_CONFIG = {
    "namespace_prefix": "dova",
    "strategies": {
        "user_preferences": {
            "description": "Stores user preferences and settings",
            "ttl_days": 365,
            "max_entries": 1000,
        },
        "semantic_facts": {
            "description": "Stores semantic facts extracted from user interactions",
            "ttl_days": 180,
            "max_entries": 5000,
        },
        "session_summaries": {
            "description": "Stores session summaries for context continuity",
            "ttl_days": 30,
            "max_entries": 100,
        },
    },
}


def get_aws_clients():
    """Get AWS clients for memory setup."""
    region = os.getenv("AWS_REGION", "us-east-1")

    return {
        "dynamodb": boto3.client("dynamodb", region_name=region),
        "s3": boto3.client("s3", region_name=region),
    }


def create_dynamodb_tables(dynamodb_client, environment: str):
    """Create DynamoDB tables for memory storage."""
    tables = [
        {
            "TableName": f"dova-{environment}-user-preferences",
            "KeySchema": [
                {"AttributeName": "user_id", "KeyType": "HASH"},
                {"AttributeName": "preference_key", "KeyType": "RANGE"},
            ],
            "AttributeDefinitions": [
                {"AttributeName": "user_id", "AttributeType": "S"},
                {"AttributeName": "preference_key", "AttributeType": "S"},
            ],
            "BillingMode": "PAY_PER_REQUEST",
            "Tags": [
                {"Key": "Project", "Value": "DOVA"},
                {"Key": "Environment", "Value": environment},
            ],
        },
        {
            "TableName": f"dova-{environment}-semantic-facts",
            "KeySchema": [
                {"AttributeName": "user_id", "KeyType": "HASH"},
                {"AttributeName": "fact_id", "KeyType": "RANGE"},
            ],
            "AttributeDefinitions": [
                {"AttributeName": "user_id", "AttributeType": "S"},
                {"AttributeName": "fact_id", "AttributeType": "S"},
            ],
            "BillingMode": "PAY_PER_REQUEST",
            "Tags": [
                {"Key": "Project", "Value": "DOVA"},
                {"Key": "Environment", "Value": environment},
            ],
        },
        {
            "TableName": f"dova-{environment}-session-summaries",
            "KeySchema": [
                {"AttributeName": "user_id", "KeyType": "HASH"},
                {"AttributeName": "session_id", "KeyType": "RANGE"},
            ],
            "AttributeDefinitions": [
                {"AttributeName": "user_id", "AttributeType": "S"},
                {"AttributeName": "session_id", "AttributeType": "S"},
            ],
            "BillingMode": "PAY_PER_REQUEST",
            "Tags": [
                {"Key": "Project", "Value": "DOVA"},
                {"Key": "Environment", "Value": environment},
            ],
            "TimeToLiveSpecification": {
                "Enabled": True,
                "AttributeName": "ttl",
            },
        },
    ]

    for table_config in tables:
        table_name = table_config["TableName"]

        try:
            # Check if table exists
            dynamodb_client.describe_table(TableName=table_name)
            print(f"Table {table_name} already exists, skipping...")

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                # Create table
                print(f"Creating table {table_name}...")

                # Extract TTL config if present
                ttl_spec = table_config.pop("TimeToLiveSpecification", None)

                dynamodb_client.create_table(**table_config)

                # Wait for table to be active
                waiter = dynamodb_client.get_waiter("table_exists")
                waiter.wait(TableName=table_name)

                # Enable TTL if specified
                if ttl_spec:
                    dynamodb_client.update_time_to_live(
                        TableName=table_name,
                        TimeToLiveSpecification=ttl_spec,
                    )

                print(f"Table {table_name} created successfully")
            else:
                raise


def create_s3_bucket(s3_client, environment: str, region: str):
    """Create S3 bucket for memory artifacts."""
    bucket_name = f"dova-{environment}-memory-{region}"

    try:
        # Check if bucket exists
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Bucket {bucket_name} already exists, skipping...")

    except ClientError as e:
        if e.response["Error"]["Code"] in ["404", "NoSuchBucket"]:
            print(f"Creating bucket {bucket_name}...")

            # Create bucket
            if region == "us-east-1":
                s3_client.create_bucket(Bucket=bucket_name)
            else:
                s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": region},
                )

            # Enable versioning
            s3_client.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={"Status": "Enabled"},
            )

            # Enable encryption
            s3_client.put_bucket_encryption(
                Bucket=bucket_name,
                ServerSideEncryptionConfiguration={
                    "Rules": [
                        {
                            "ApplyServerSideEncryptionByDefault": {
                                "SSEAlgorithm": "AES256",
                            },
                        },
                    ],
                },
            )

            # Block public access
            s3_client.put_public_access_block(
                Bucket=bucket_name,
                PublicAccessBlockConfiguration={
                    "BlockPublicAcls": True,
                    "IgnorePublicAcls": True,
                    "BlockPublicPolicy": True,
                    "RestrictPublicBuckets": True,
                },
            )

            # Add lifecycle policy for session summaries
            s3_client.put_bucket_lifecycle_configuration(
                Bucket=bucket_name,
                LifecycleConfiguration={
                    "Rules": [
                        {
                            "ID": "expire-session-summaries",
                            "Filter": {"Prefix": "session-summaries/"},
                            "Status": "Enabled",
                            "Expiration": {"Days": 30},
                        },
                        {
                            "ID": "expire-semantic-facts",
                            "Filter": {"Prefix": "semantic-facts/"},
                            "Status": "Enabled",
                            "Expiration": {"Days": 180},
                        },
                    ],
                },
            )

            print(f"Bucket {bucket_name} created successfully")
        else:
            raise


def create_memory_config_file(environment: str):
    """Create memory configuration file for the application."""
    config = {
        "version": "1.0",
        "created_at": datetime.utcnow().isoformat(),
        "environment": environment,
        "strategies": MEMORY_CONFIG["strategies"],
        "namespaces": {
            "user_preferences": f"dova-{environment}-user-preferences",
            "semantic_facts": f"dova-{environment}-semantic-facts",
            "session_summaries": f"dova-{environment}-session-summaries",
        },
    }

    # Save to local file
    config_path = f"memory-config-{environment}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Memory configuration saved to {config_path}")

    return config


def seed_sample_data(dynamodb_client, environment: str):
    """Seed sample data for testing."""
    sample_user = {
        "user_id": {"S": "test-user-001"},
        "preference_key": {"S": "interests"},
        "value": {"S": json.dumps(["machine learning", "NLP", "transformers"])},
        "created_at": {"S": datetime.utcnow().isoformat()},
        "updated_at": {"S": datetime.utcnow().isoformat()},
    }

    try:
        dynamodb_client.put_item(
            TableName=f"dova-{environment}-user-preferences",
            Item=sample_user,
            ConditionExpression="attribute_not_exists(user_id)",
        )
        print("Sample user data seeded")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            print("Sample user data already exists, skipping...")
        else:
            raise


def main():
    """Main function to set up memory."""
    environment = os.getenv("ENVIRONMENT", "dev")
    region = os.getenv("AWS_REGION", "us-east-1")

    print(f"Setting up DOVA memory for environment: {environment}")
    print(f"AWS Region: {region}")
    print("")

    try:
        clients = get_aws_clients()

        # Create DynamoDB tables
        print("Creating DynamoDB tables...")
        create_dynamodb_tables(clients["dynamodb"], environment)
        print("")

        # Create S3 bucket
        print("Creating S3 bucket...")
        create_s3_bucket(clients["s3"], environment, region)
        print("")

        # Create config file
        print("Creating memory configuration...")
        config = create_memory_config_file(environment)
        print("")

        # Seed sample data (dev only)
        if environment == "dev":
            print("Seeding sample data...")
            seed_sample_data(clients["dynamodb"], environment)
            print("")

        print("=" * 50)
        print("Memory setup complete!")
        print("=" * 50)
        print("")
        print("Resources created:")
        print(f"  - DynamoDB: dova-{environment}-user-preferences")
        print(f"  - DynamoDB: dova-{environment}-semantic-facts")
        print(f"  - DynamoDB: dova-{environment}-session-summaries")
        print(f"  - S3: dova-{environment}-memory-{region}")
        print("")

    except ClientError as e:
        print(f"AWS Error: {e.response['Error']['Message']}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
