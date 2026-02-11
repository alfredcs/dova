#!/usr/bin/env python3
"""
Script to add AWS MCP servers from awslabs/mcp repository to Dova.

Usage:
    python scripts/add_aws_mcp_servers.py [--all] [--list] [--interactive]

Options:
    --all         Add all available AWS MCP servers
    --list        List all available servers without adding
    --interactive Interactively select servers to add (default)
"""

import argparse
import json
import sys
from pathlib import Path

# All available AWS MCP servers from awslabs/mcp repository
AWS_MCP_SERVERS = {
    # Core/Essential
    "aws-documentation": {
        "description": "AWS documentation and best practices search",
        "category": "Documentation",
        "env": {"AWS_DOCUMENTATION_PARTITION": "aws"},
    },
    "aws-api": {
        "description": "Generic AWS API access via MCP",
        "category": "Core",
    },
    "core": {
        "description": "Core AWS MCP server utilities",
        "category": "Core",
    },
    # Compute
    "ecs": {
        "description": "Amazon ECS container orchestration",
        "category": "Compute",
    },
    "eks": {
        "description": "Amazon EKS Kubernetes service",
        "category": "Compute",
    },
    "lambda-tool": {
        "description": "AWS Lambda function management",
        "category": "Compute",
    },
    "aws-serverless": {
        "description": "AWS Serverless services (Lambda, API Gateway, etc.)",
        "category": "Compute",
    },
    "stepfunctions-tool": {
        "description": "AWS Step Functions workflow management",
        "category": "Compute",
    },
    # Databases
    "dynamodb": {
        "description": "Amazon DynamoDB NoSQL database",
        "category": "Database",
    },
    "documentdb": {
        "description": "Amazon DocumentDB (MongoDB-compatible)",
        "category": "Database",
    },
    "aurora-dsql": {
        "description": "Amazon Aurora DSQL serverless database",
        "category": "Database",
    },
    "amazon-neptune": {
        "description": "Amazon Neptune graph database",
        "category": "Database",
    },
    "amazon-keyspaces": {
        "description": "Amazon Keyspaces (Cassandra-compatible)",
        "category": "Database",
    },
    "mysql": {
        "description": "MySQL database on AWS",
        "category": "Database",
    },
    "postgres": {
        "description": "PostgreSQL database on AWS",
        "category": "Database",
    },
    "redshift": {
        "description": "Amazon Redshift data warehouse",
        "category": "Database",
    },
    "s3-tables": {
        "description": "Amazon S3 Tables (Apache Iceberg)",
        "category": "Database",
    },
    "timestream-for-influxdb": {
        "description": "Amazon Timestream for InfluxDB",
        "category": "Database",
    },
    # Caching
    "elasticache": {
        "description": "Amazon ElastiCache (Redis/Memcached)",
        "category": "Caching",
    },
    "valkey": {
        "description": "Valkey (Redis-compatible) on AWS",
        "category": "Caching",
    },
    "memcached": {
        "description": "Memcached on AWS",
        "category": "Caching",
    },
    # AI/ML
    "bedrock-kb-retrieval": {
        "description": "Amazon Bedrock Knowledge Base retrieval",
        "category": "AI/ML",
    },
    "amazon-bedrock-agentcore": {
        "description": "Amazon Bedrock AgentCore",
        "category": "AI/ML",
    },
    "aws-bedrock-custom-model-import": {
        "description": "Import custom models to Amazon Bedrock",
        "category": "AI/ML",
    },
    "aws-bedrock-data-automation": {
        "description": "Bedrock data automation pipelines",
        "category": "AI/ML",
    },
    "sagemaker-ai": {
        "description": "Amazon SageMaker AI services",
        "category": "AI/ML",
    },
    "sagemaker-unified-studio-spark-troubleshooting": {
        "description": "SageMaker Spark troubleshooting tools",
        "category": "AI/ML",
    },
    "sagemaker-unified-studio-spark-upgrade": {
        "description": "SageMaker Spark upgrade assistant",
        "category": "AI/ML",
    },
    "nova-canvas": {
        "description": "Amazon Nova Canvas image generation",
        "category": "AI/ML",
    },
    "syntheticdata": {
        "description": "Synthetic data generation on AWS",
        "category": "AI/ML",
    },
    # Search & Knowledge
    "amazon-kendra-index": {
        "description": "Amazon Kendra enterprise search",
        "category": "Search",
    },
    "amazon-qbusiness-anonymous": {
        "description": "Amazon Q Business (anonymous access)",
        "category": "Search",
    },
    "amazon-qindex": {
        "description": "Amazon Q Index services",
        "category": "Search",
    },
    "aws-knowledge": {
        "description": "AWS Knowledge Base services",
        "category": "Search",
    },
    # Monitoring & Operations
    "cloudwatch": {
        "description": "Amazon CloudWatch monitoring",
        "category": "Monitoring",
    },
    "cloudwatch-applicationsignals": {
        "description": "CloudWatch Application Signals",
        "category": "Monitoring",
    },
    "cloudwatch-appsignals": {
        "description": "CloudWatch AppSignals observability",
        "category": "Monitoring",
    },
    "cloudtrail": {
        "description": "AWS CloudTrail audit logs",
        "category": "Monitoring",
    },
    "prometheus": {
        "description": "Amazon Managed Prometheus",
        "category": "Monitoring",
    },
    "aws-support": {
        "description": "AWS Support case management",
        "category": "Operations",
    },
    # Infrastructure as Code
    "cdk": {
        "description": "AWS CDK infrastructure as code",
        "category": "IaC",
    },
    "cfn": {
        "description": "AWS CloudFormation templates",
        "category": "IaC",
    },
    "terraform": {
        "description": "Terraform on AWS",
        "category": "IaC",
    },
    "aws-iac": {
        "description": "AWS Infrastructure as Code tools",
        "category": "IaC",
    },
    # Networking
    "aws-network": {
        "description": "AWS networking services (VPC, etc.)",
        "category": "Networking",
    },
    "aws-location": {
        "description": "Amazon Location Service (maps, geolocation)",
        "category": "Networking",
    },
    # Messaging
    "amazon-sns-sqs": {
        "description": "Amazon SNS/SQS messaging",
        "category": "Messaging",
    },
    "amazon-mq": {
        "description": "Amazon MQ message broker",
        "category": "Messaging",
    },
    "aws-msk": {
        "description": "Amazon MSK (Kafka)",
        "category": "Messaging",
    },
    # Security & IAM
    "iam": {
        "description": "AWS IAM identity management",
        "category": "Security",
    },
    "well-architected-security": {
        "description": "AWS Well-Architected security best practices",
        "category": "Security",
    },
    # Cost Management
    "billing-cost-management": {
        "description": "AWS Billing and Cost Management",
        "category": "Cost",
    },
    "cost-explorer": {
        "description": "AWS Cost Explorer analysis",
        "category": "Cost",
    },
    "aws-pricing": {
        "description": "AWS Pricing information",
        "category": "Cost",
    },
    # Healthcare
    "aws-healthomics": {
        "description": "AWS HealthOmics genomics data",
        "category": "Healthcare",
    },
    "healthlake": {
        "description": "Amazon HealthLake FHIR data store",
        "category": "Healthcare",
    },
    # Development Tools
    "aws-appsync": {
        "description": "AWS AppSync GraphQL APIs",
        "category": "Development",
    },
    "aws-dataprocessing": {
        "description": "AWS data processing services",
        "category": "Development",
    },
    "aws-diagram": {
        "description": "AWS architecture diagram generation",
        "category": "Development",
    },
    "code-doc-gen": {
        "description": "Code documentation generation",
        "category": "Development",
    },
    "document-loader": {
        "description": "Document loading and processing",
        "category": "Development",
    },
    "git-repo-research": {
        "description": "Git repository research tools",
        "category": "Development",
    },
    "openapi": {
        "description": "OpenAPI specification tools",
        "category": "Development",
    },
    "frontend": {
        "description": "Frontend development tools",
        "category": "Development",
    },
    "finch": {
        "description": "Finch container development",
        "category": "Development",
    },
    # IoT
    "aws-iot-sitewise": {
        "description": "AWS IoT SiteWise industrial data",
        "category": "IoT",
    },
    # Other
    "ccapi": {
        "description": "Cloud Control API access",
        "category": "Other",
    },
    "mcp-lambda-handler": {
        "description": "MCP Lambda handler utilities",
        "category": "Other",
    },
}


def get_dova_config_path() -> Path:
    """Get the path to ~/.dova.json."""
    return Path.home() / ".dova.json"


def load_dova_config() -> dict:
    """Load existing Dova configuration."""
    config_path = get_dova_config_path()
    if not config_path.exists():
        return {}
    try:
        with open(config_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_dova_config(config: dict) -> None:
    """Save Dova configuration."""
    config_path = get_dova_config_path()
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_path}")


def get_server_package_name(server_name: str) -> str:
    """Get the full package name for an AWS MCP server."""
    return f"awslabs.{server_name}-mcp-server"


def create_server_config(server_name: str, server_info: dict) -> dict:
    """Create MCP server configuration for Dova."""
    package_name = get_server_package_name(server_name)
    config = {
        "command": "uvx",
        "args": [f"{package_name}@latest"],
        "env": {"FASTMCP_LOG_LEVEL": "ERROR"},
    }
    # Add any server-specific environment variables
    if "env" in server_info:
        config["env"].update(server_info["env"])
    return config


def list_servers() -> None:
    """List all available AWS MCP servers."""
    # Group by category
    categories: dict[str, list[tuple[str, dict]]] = {}
    for name, info in sorted(AWS_MCP_SERVERS.items()):
        cat = info.get("category", "Other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, info))

    print("\n" + "=" * 70)
    print("Available AWS MCP Servers from awslabs/mcp")
    print("=" * 70)

    for category in sorted(categories.keys()):
        print(f"\n[{category}]")
        for name, info in categories[category]:
            package = get_server_package_name(name)
            print(f"  {name:40} - {info['description']}")

    print(f"\nTotal: {len(AWS_MCP_SERVERS)} servers available")
    print("\nTo add servers, run: python scripts/add_aws_mcp_servers.py --interactive")


def add_all_servers() -> None:
    """Add all AWS MCP servers to Dova config."""
    config = load_dova_config()
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    added = 0
    for name, info in AWS_MCP_SERVERS.items():
        server_key = get_server_package_name(name)
        if server_key not in config["mcpServers"]:
            config["mcpServers"][server_key] = create_server_config(name, info)
            print(f"  Added: {server_key}")
            added += 1
        else:
            print(f"  Exists: {server_key}")

    save_dova_config(config)
    print(f"\nAdded {added} new servers, {len(AWS_MCP_SERVERS) - added} already existed")


def interactive_select() -> None:
    """Interactively select servers to add."""
    config = load_dova_config()
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Group by category for easier selection
    categories: dict[str, list[tuple[str, dict]]] = {}
    for name, info in sorted(AWS_MCP_SERVERS.items()):
        cat = info.get("category", "Other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, info))

    print("\n" + "=" * 70)
    print("AWS MCP Server Selection")
    print("=" * 70)
    print("\nSelect categories or individual servers to add.")
    print("Enter 'q' to finish, 'a' for all, or category/server numbers.\n")

    # Create numbered list
    items: list[tuple[str, str | None, dict | None]] = []  # (label, server_name, info)

    for category in sorted(categories.keys()):
        items.append((f"[Category: {category}]", None, None))
        for name, info in categories[category]:
            package = get_server_package_name(name)
            exists = "✓" if package in config["mcpServers"] else " "
            items.append((f"  {exists} {name}: {info['description'][:50]}", name, info))

    # Display with numbers
    for i, (label, _, _) in enumerate(items):
        if label.startswith("[Category"):
            print(f"\n{label}")
        else:
            print(f"  {i:3}. {label}")

    print("\n" + "-" * 70)
    print("Commands: [number] to toggle, [a]ll, [c]ategory name, [q]uit and save")

    selected: set[str] = set()

    while True:
        try:
            choice = input("\nSelect (or 'q' to save & quit): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            return

        if choice == "q":
            break
        elif choice == "a":
            # Add all
            selected = set(AWS_MCP_SERVERS.keys())
            print(f"Selected all {len(selected)} servers")
        elif choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(items):
                _, server_name, _ = items[idx]
                if server_name:
                    if server_name in selected:
                        selected.discard(server_name)
                        print(f"Deselected: {server_name}")
                    else:
                        selected.add(server_name)
                        print(f"Selected: {server_name}")
                else:
                    print("That's a category header, select individual servers.")
        elif choice.startswith("c "):
            # Select by category
            cat_name = choice[2:].strip()
            matched = False
            for category, servers in categories.items():
                if cat_name.lower() in category.lower():
                    for name, _ in servers:
                        selected.add(name)
                    print(f"Selected all {len(servers)} servers in {category}")
                    matched = True
                    break
            if not matched:
                print(f"Category '{cat_name}' not found")
        else:
            print("Invalid input. Enter a number, 'a' for all, 'c <category>', or 'q' to quit.")

    # Add selected servers
    if selected:
        added = 0
        for name in selected:
            server_key = get_server_package_name(name)
            if server_key not in config["mcpServers"]:
                config["mcpServers"][server_key] = create_server_config(
                    name, AWS_MCP_SERVERS[name]
                )
                print(f"  Added: {server_key}")
                added += 1
            else:
                print(f"  Exists: {server_key}")

        save_dova_config(config)
        print(f"\nAdded {added} servers to ~/.dova.json")
    else:
        print("No servers selected.")


def add_recommended() -> None:
    """Add recommended/essential AWS MCP servers."""
    recommended = [
        "aws-documentation",
        "dynamodb",
        "aws-api",
        "cloudwatch",
        "bedrock-kb-retrieval",
        "iam",
        "cdk",
        "aws-pricing",
    ]

    config = load_dova_config()
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    print("\nAdding recommended AWS MCP servers:")
    added = 0
    for name in recommended:
        if name in AWS_MCP_SERVERS:
            server_key = get_server_package_name(name)
            if server_key not in config["mcpServers"]:
                config["mcpServers"][server_key] = create_server_config(
                    name, AWS_MCP_SERVERS[name]
                )
                print(f"  Added: {server_key}")
                added += 1
            else:
                print(f"  Exists: {server_key}")

    save_dova_config(config)
    print(f"\nAdded {added} recommended servers")


def main():
    parser = argparse.ArgumentParser(
        description="Add AWS MCP servers from awslabs/mcp to Dova"
    )
    parser.add_argument(
        "--all", action="store_true", help="Add all available AWS MCP servers"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available servers"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactively select servers (default)",
    )
    parser.add_argument(
        "--recommended",
        "-r",
        action="store_true",
        help="Add recommended essential servers",
    )

    args = parser.parse_args()

    print("AWS MCP Server Manager for Dova")
    print("Source: https://github.com/awslabs/mcp")

    if args.list:
        list_servers()
    elif args.all:
        print("\nAdding all AWS MCP servers...")
        add_all_servers()
    elif args.recommended:
        add_recommended()
    else:
        # Default to interactive
        interactive_select()


if __name__ == "__main__":
    main()
