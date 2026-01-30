"""
DOVA Gateway Lambda Function.

Handles API Gateway requests and routes to DOVA agents.
"""

import json
import os
from typing import Any

import boto3

# Environment variables
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-20250514-v1:0"
)
PROFILE_TABLE_NAME = os.environ.get("PROFILE_TABLE_NAME", "")
CACHE_BUCKET_NAME = os.environ.get("CACHE_BUCKET_NAME", "")

# AWS clients
bedrock_client = boto3.client("bedrock-runtime")
dynamodb = boto3.resource("dynamodb")


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Lambda handler for API Gateway requests.

    Routes requests to appropriate DOVA functionality.
    """
    print(f"Event: {json.dumps(event)}")

    # Extract request details
    http_method = event.get("httpMethod", "GET")
    path = event.get("path", "/")
    body = json.loads(event.get("body", "{}")) if event.get("body") else {}
    headers = event.get("headers", {})

    # Get user info from authorizer
    claims = event.get("requestContext", {}).get("authorizer", {}).get("claims", {})
    user_id = claims.get("sub", "anonymous")

    try:
        # Route based on path
        if path.startswith("/api/v1/research"):
            result = handle_research(body, user_id)
        elif path.startswith("/api/v1/search"):
            source = path.split("/")[-1] if "/" in path else "all"
            result = handle_search(source, body, user_id)
        elif path.startswith("/api/v1/profile"):
            if http_method == "GET":
                result = handle_get_profile(user_id)
            else:
                result = handle_update_profile(body, user_id)
        elif path.startswith("/api/v1/validate"):
            result = handle_validation(body, user_id)
        else:
            result = {"error": "Unknown endpoint"}

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(result),
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"error": str(e)}),
        }


def handle_research(body: dict[str, Any], user_id: str) -> dict[str, Any]:
    """Handle research request."""
    query = body.get("query", "")
    if not query:
        return {"error": "No query provided"}

    # Call Bedrock for research synthesis
    prompt = f"""You are a research assistant. Analyze the following research query and provide insights:

Query: {query}

Provide:
1. Key topics to research
2. Suggested sources (ArXiv, GitHub, HuggingFace)
3. Initial research direction"""

    response = call_bedrock(prompt)

    return {
        "query": query,
        "status": "completed",
        "summary": response,
        "papers": [],
        "repositories": [],
        "models": [],
    }


def handle_search(source: str, body: dict[str, Any], user_id: str) -> dict[str, Any]:
    """Handle search request for a specific source."""
    query = body.get("query", "")
    return {
        "source": source,
        "query": query,
        "results": [],
        "message": f"Search endpoint for {source} - requires MCP integration",
    }


def handle_get_profile(user_id: str) -> dict[str, Any]:
    """Get user profile from DynamoDB."""
    if not PROFILE_TABLE_NAME:
        return {"user_id": user_id, "preferences": {}, "message": "Table not configured"}

    table = dynamodb.Table(PROFILE_TABLE_NAME)
    response = table.get_item(Key={"user_id": user_id})

    if "Item" in response:
        return response["Item"]

    return {
        "user_id": user_id,
        "preferences": {
            "interests": [],
            "preferred_sources": ["arxiv", "github", "huggingface"],
            "expertise_level": "intermediate",
        },
    }


def handle_update_profile(body: dict[str, Any], user_id: str) -> dict[str, Any]:
    """Update user profile in DynamoDB."""
    if not PROFILE_TABLE_NAME:
        return {"status": "error", "message": "Table not configured"}

    table = dynamodb.Table(PROFILE_TABLE_NAME)
    table.update_item(
        Key={"user_id": user_id},
        UpdateExpression="SET preferences = :prefs",
        ExpressionAttributeValues={":prefs": body.get("preferences", {})},
    )

    return {"status": "updated", "user_id": user_id}


def handle_validation(body: dict[str, Any], user_id: str) -> dict[str, Any]:
    """Handle code validation request."""
    code = body.get("code", "")
    language = body.get("language", "python")

    if not code:
        return {"error": "No code provided"}

    # Use Bedrock for code analysis
    prompt = f"""Analyze this {language} code for quality and security issues:

```{language}
{code}
```

Provide:
1. Overall quality score (0-100)
2. List of issues found
3. Suggestions for improvement"""

    response = call_bedrock(prompt)

    return {
        "status": "completed",
        "analysis": response,
        "language": language,
    }


def call_bedrock(prompt: str) -> str:
    """Call Bedrock for LLM inference."""
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }

    response = bedrock_client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]
