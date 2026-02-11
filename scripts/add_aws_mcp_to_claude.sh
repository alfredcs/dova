#!/bin/bash
# Script to add AWS MCP servers from ~/.dova.json to Claude Code
# Generated from the mcpServers in ~/.dova.json

set -e

echo "Adding AWS MCP servers to Claude Code..."
echo "==========================================="

# AWS Documentation MCP Server
echo "Adding: awslabs-aws-documentation-mcp-server"
claude mcp add awslabs-aws-documentation-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -e AWS_DOCUMENTATION_PARTITION=aws \
  -- uvx awslabs.aws-documentation-mcp-server@latest

# AWS API MCP Server
echo "Adding: awslabs-aws-api-mcp-server"
claude mcp add awslabs-aws-api-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.aws-api-mcp-server@latest

# Core MCP Server
echo "Adding: awslabs-core-mcp-server"
claude mcp add awslabs-core-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.core-mcp-server@latest

# ECS MCP Server
echo "Adding: awslabs-ecs-mcp-server"
claude mcp add awslabs-ecs-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.ecs-mcp-server@latest

# EKS MCP Server
echo "Adding: awslabs-eks-mcp-server"
claude mcp add awslabs-eks-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.eks-mcp-server@latest

# Lambda Tool MCP Server
echo "Adding: awslabs-lambda-tool-mcp-server"
claude mcp add awslabs-lambda-tool-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.lambda-tool-mcp-server@latest

# AWS Serverless MCP Server
echo "Adding: awslabs-aws-serverless-mcp-server"
claude mcp add awslabs-aws-serverless-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.aws-serverless-mcp-server@latest

# Step Functions Tool MCP Server
echo "Adding: awslabs-stepfunctions-tool-mcp-server"
claude mcp add awslabs-stepfunctions-tool-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.stepfunctions-tool-mcp-server@latest

# DynamoDB MCP Server
echo "Adding: awslabs-dynamodb-mcp-server"
claude mcp add awslabs-dynamodb-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.dynamodb-mcp-server@latest

# DocumentDB MCP Server
echo "Adding: awslabs-documentdb-mcp-server"
claude mcp add awslabs-documentdb-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.documentdb-mcp-server@latest

# Aurora DSQL MCP Server
echo "Adding: awslabs-aurora-dsql-mcp-server"
claude mcp add awslabs-aurora-dsql-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.aurora-dsql-mcp-server@latest

# Amazon Neptune MCP Server
echo "Adding: awslabs-amazon-neptune-mcp-server"
claude mcp add awslabs-amazon-neptune-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.amazon-neptune-mcp-server@latest

# Amazon Keyspaces MCP Server
echo "Adding: awslabs-amazon-keyspaces-mcp-server"
claude mcp add awslabs-amazon-keyspaces-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.amazon-keyspaces-mcp-server@latest

# MySQL MCP Server
echo "Adding: awslabs-mysql-mcp-server"
claude mcp add awslabs-mysql-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.mysql-mcp-server@latest

# PostgreSQL MCP Server
echo "Adding: awslabs-postgres-mcp-server"
claude mcp add awslabs-postgres-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.postgres-mcp-server@latest

# Redshift MCP Server
echo "Adding: awslabs-redshift-mcp-server"
claude mcp add awslabs-redshift-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.redshift-mcp-server@latest

# S3 Tables MCP Server
echo "Adding: awslabs-s3-tables-mcp-server"
claude mcp add awslabs-s3-tables-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.s3-tables-mcp-server@latest

# Timestream for InfluxDB MCP Server
echo "Adding: awslabs-timestream-for-influxdb-mcp-server"
claude mcp add awslabs-timestream-for-influxdb-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.timestream-for-influxdb-mcp-server@latest

# ElastiCache MCP Server
echo "Adding: awslabs-elasticache-mcp-server"
claude mcp add awslabs-elasticache-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.elasticache-mcp-server@latest

# Valkey MCP Server
echo "Adding: awslabs-valkey-mcp-server"
claude mcp add awslabs-valkey-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.valkey-mcp-server@latest

# Memcached MCP Server
echo "Adding: awslabs-memcached-mcp-server"
claude mcp add awslabs-memcached-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.memcached-mcp-server@latest

# Bedrock KB Retrieval MCP Server
echo "Adding: awslabs-bedrock-kb-retrieval-mcp-server"
claude mcp add awslabs-bedrock-kb-retrieval-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.bedrock-kb-retrieval-mcp-server@latest

# Amazon Bedrock AgentCore MCP Server
echo "Adding: awslabs-amazon-bedrock-agentcore-mcp-server"
claude mcp add awslabs-amazon-bedrock-agentcore-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.amazon-bedrock-agentcore-mcp-server@latest

# AWS Bedrock Custom Model Import MCP Server
echo "Adding: awslabs-aws-bedrock-custom-model-import-mcp-server"
claude mcp add awslabs-aws-bedrock-custom-model-import-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.aws-bedrock-custom-model-import-mcp-server@latest

# AWS Bedrock Data Automation MCP Server
echo "Adding: awslabs-aws-bedrock-data-automation-mcp-server"
claude mcp add awslabs-aws-bedrock-data-automation-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.aws-bedrock-data-automation-mcp-server@latest

# SageMaker AI MCP Server
echo "Adding: awslabs-sagemaker-ai-mcp-server"
claude mcp add awslabs-sagemaker-ai-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.sagemaker-ai-mcp-server@latest

# SageMaker Unified Studio Spark Troubleshooting MCP Server
echo "Adding: awslabs-sagemaker-unified-studio-spark-troubleshooting-mcp-server"
claude mcp add awslabs-sagemaker-unified-studio-spark-troubleshooting-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.sagemaker-unified-studio-spark-troubleshooting-mcp-server@latest

# SageMaker Unified Studio Spark Upgrade MCP Server
echo "Adding: awslabs-sagemaker-unified-studio-spark-upgrade-mcp-server"
claude mcp add awslabs-sagemaker-unified-studio-spark-upgrade-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.sagemaker-unified-studio-spark-upgrade-mcp-server@latest

# Nova Canvas MCP Server
echo "Adding: awslabs-nova-canvas-mcp-server"
claude mcp add awslabs-nova-canvas-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.nova-canvas-mcp-server@latest

# Synthetic Data MCP Server
echo "Adding: awslabs-syntheticdata-mcp-server"
claude mcp add awslabs-syntheticdata-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.syntheticdata-mcp-server@latest

# Amazon Kendra Index MCP Server
echo "Adding: awslabs-amazon-kendra-index-mcp-server"
claude mcp add awslabs-amazon-kendra-index-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.amazon-kendra-index-mcp-server@latest

# Amazon Q Business Anonymous MCP Server
echo "Adding: awslabs-amazon-qbusiness-anonymous-mcp-server"
claude mcp add awslabs-amazon-qbusiness-anonymous-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.amazon-qbusiness-anonymous-mcp-server@latest

# Amazon Q Index MCP Server
echo "Adding: awslabs-amazon-qindex-mcp-server"
claude mcp add awslabs-amazon-qindex-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.amazon-qindex-mcp-server@latest

# AWS Knowledge MCP Server
echo "Adding: awslabs-aws-knowledge-mcp-server"
claude mcp add awslabs-aws-knowledge-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.aws-knowledge-mcp-server@latest

# CloudWatch MCP Server
echo "Adding: awslabs-cloudwatch-mcp-server"
claude mcp add awslabs-cloudwatch-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.cloudwatch-mcp-server@latest

# CloudWatch Application Signals MCP Server
echo "Adding: awslabs-cloudwatch-applicationsignals-mcp-server"
claude mcp add awslabs-cloudwatch-applicationsignals-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.cloudwatch-applicationsignals-mcp-server@latest

# CloudWatch AppSignals MCP Server
echo "Adding: awslabs-cloudwatch-appsignals-mcp-server"
claude mcp add awslabs-cloudwatch-appsignals-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.cloudwatch-appsignals-mcp-server@latest

# CloudTrail MCP Server
echo "Adding: awslabs-cloudtrail-mcp-server"
claude mcp add awslabs-cloudtrail-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.cloudtrail-mcp-server@latest

# Prometheus MCP Server
echo "Adding: awslabs-prometheus-mcp-server"
claude mcp add awslabs-prometheus-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.prometheus-mcp-server@latest

# AWS Support MCP Server
echo "Adding: awslabs-aws-support-mcp-server"
claude mcp add awslabs-aws-support-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.aws-support-mcp-server@latest

# CDK MCP Server
echo "Adding: awslabs-cdk-mcp-server"
claude mcp add awslabs-cdk-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.cdk-mcp-server@latest

# CloudFormation MCP Server
echo "Adding: awslabs-cfn-mcp-server"
claude mcp add awslabs-cfn-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.cfn-mcp-server@latest

# Terraform MCP Server
echo "Adding: awslabs-terraform-mcp-server"
claude mcp add awslabs-terraform-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.terraform-mcp-server@latest

# AWS IaC MCP Server
echo "Adding: awslabs-aws-iac-mcp-server"
claude mcp add awslabs-aws-iac-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.aws-iac-mcp-server@latest

# AWS Network MCP Server
echo "Adding: awslabs-aws-network-mcp-server"
claude mcp add awslabs-aws-network-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.aws-network-mcp-server@latest

# AWS Location MCP Server
echo "Adding: awslabs-aws-location-mcp-server"
claude mcp add awslabs-aws-location-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.aws-location-mcp-server@latest

# Amazon SNS/SQS MCP Server
echo "Adding: awslabs-amazon-sns-sqs-mcp-server"
claude mcp add awslabs-amazon-sns-sqs-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.amazon-sns-sqs-mcp-server@latest

# Amazon MQ MCP Server
echo "Adding: awslabs-amazon-mq-mcp-server"
claude mcp add awslabs-amazon-mq-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.amazon-mq-mcp-server@latest

# AWS MSK MCP Server
echo "Adding: awslabs-aws-msk-mcp-server"
claude mcp add awslabs-aws-msk-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.aws-msk-mcp-server@latest

# IAM MCP Server
echo "Adding: awslabs-iam-mcp-server"
claude mcp add awslabs-iam-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.iam-mcp-server@latest

# Well-Architected Security MCP Server
echo "Adding: awslabs-well-architected-security-mcp-server"
claude mcp add awslabs-well-architected-security-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.well-architected-security-mcp-server@latest

# Billing Cost Management MCP Server
echo "Adding: awslabs-billing-cost-management-mcp-server"
claude mcp add awslabs-billing-cost-management-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.billing-cost-management-mcp-server@latest

# Cost Explorer MCP Server
echo "Adding: awslabs-cost-explorer-mcp-server"
claude mcp add awslabs-cost-explorer-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.cost-explorer-mcp-server@latest

# AWS Pricing MCP Server
echo "Adding: awslabs-aws-pricing-mcp-server"
claude mcp add awslabs-aws-pricing-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.aws-pricing-mcp-server@latest

# AWS HealthOmics MCP Server
echo "Adding: awslabs-aws-healthomics-mcp-server"
claude mcp add awslabs-aws-healthomics-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.aws-healthomics-mcp-server@latest

# HealthLake MCP Server
echo "Adding: awslabs-healthlake-mcp-server"
claude mcp add awslabs-healthlake-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.healthlake-mcp-server@latest

# AWS AppSync MCP Server
echo "Adding: awslabs-aws-appsync-mcp-server"
claude mcp add awslabs-aws-appsync-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.aws-appsync-mcp-server@latest

# AWS Data Processing MCP Server
echo "Adding: awslabs-aws-dataprocessing-mcp-server"
claude mcp add awslabs-aws-dataprocessing-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.aws-dataprocessing-mcp-server@latest

# AWS Diagram MCP Server
echo "Adding: awslabs-aws-diagram-mcp-server"
claude mcp add awslabs-aws-diagram-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.aws-diagram-mcp-server@latest

# Code Doc Gen MCP Server
echo "Adding: awslabs-code-doc-gen-mcp-server"
claude mcp add awslabs-code-doc-gen-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.code-doc-gen-mcp-server@latest

# Document Loader MCP Server
echo "Adding: awslabs-document-loader-mcp-server"
claude mcp add awslabs-document-loader-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.document-loader-mcp-server@latest

# Git Repo Research MCP Server
echo "Adding: awslabs-git-repo-research-mcp-server"
claude mcp add awslabs-git-repo-research-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.git-repo-research-mcp-server@latest

# OpenAPI MCP Server
echo "Adding: awslabs-openapi-mcp-server"
claude mcp add awslabs-openapi-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.openapi-mcp-server@latest

# Frontend MCP Server
echo "Adding: awslabs-frontend-mcp-server"
claude mcp add awslabs-frontend-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.frontend-mcp-server@latest

# Finch MCP Server
echo "Adding: awslabs-finch-mcp-server"
claude mcp add awslabs-finch-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.finch-mcp-server@latest

# AWS IoT SiteWise MCP Server
echo "Adding: awslabs-aws-iot-sitewise-mcp-server"
claude mcp add awslabs-aws-iot-sitewise-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.aws-iot-sitewise-mcp-server@latest

# CCAPI MCP Server
echo "Adding: awslabs-ccapi-mcp-server"
claude mcp add awslabs-ccapi-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.ccapi-mcp-server@latest

# MCP Lambda Handler MCP Server
echo "Adding: awslabs-mcp-lambda-handler-mcp-server"
claude mcp add awslabs-mcp-lambda-handler-mcp-server \
  -e FASTMCP_LOG_LEVEL=ERROR \
  -- uvx awslabs.mcp-lambda-handler-mcp-server@latest

echo ""
echo "==========================================="
echo "Done! Added 64 AWS MCP servers to Claude Code."
echo "Run 'claude mcp list' to verify."
