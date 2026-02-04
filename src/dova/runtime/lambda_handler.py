"""Lambda handler for DOVA AgentCore.

This module provides the AWS Lambda entry point for DOVA.
It handles API Gateway events, processes requests through the agent,
and returns properly formatted responses.
"""

import asyncio
import json
import os
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def handler(event: dict, context: Any) -> dict:
    """AWS Lambda handler for DOVA.

    Handles API Gateway proxy integration events and routes to the agent.

    Args:
        event: API Gateway event containing request data
        context: Lambda context object

    Returns:
        API Gateway response dict with statusCode, headers, and body
    """
    # Set up logging with request context
    request_id = context.aws_request_id if context else "local"
    log = logger.bind(request_id=request_id)

    log.info(
        "lambda_invocation",
        path=event.get("path"),
        method=event.get("httpMethod"),
    )

    try:
        # Parse request body
        body = _parse_body(event)

        if not body:
            return _error_response(400, "Request body is required")

        # Extract prompt from body
        prompt = body.get("prompt")
        if not prompt:
            return _error_response(400, "Missing 'prompt' field in request body")

        # Build payload for agent
        payload = {
            "prompt": prompt,
            "userId": body.get("userId", "anonymous"),
            "runtimeSessionId": body.get("sessionId", request_id),
        }

        # Optional fields
        if "sources" in body:
            payload["sources"] = body["sources"]
        if "reasoning_mode" in body:
            payload["reasoning_mode"] = body["reasoning_mode"]

        # Run the agent
        response_text = asyncio.get_event_loop().run_until_complete(
            _run_agent(payload)
        )

        log.info("lambda_success", response_length=len(response_text))

        return _success_response({"response": response_text})

    except json.JSONDecodeError as e:
        log.error("json_parse_error", error=str(e))
        return _error_response(400, f"Invalid JSON: {str(e)}")

    except Exception as e:
        log.exception("lambda_error", error=str(e))
        return _error_response(500, f"Internal error: {str(e)}")


async def _run_agent(payload: dict) -> str:
    """Run the DOVA agent and collect response.

    Since Lambda/API Gateway doesn't support true streaming,
    we collect the full response before returning.

    Args:
        payload: Agent payload with prompt and context

    Returns:
        Complete response text
    """
    from dova.runtime.agentcore_app import agent_stream

    chunks = []
    async for chunk in agent_stream(payload):
        chunks.append(chunk)

    return "".join(chunks)


def _parse_body(event: dict) -> dict | None:
    """Parse the request body from the event.

    Handles both raw JSON and base64-encoded bodies.

    Args:
        event: API Gateway event

    Returns:
        Parsed body dict or None
    """
    body = event.get("body")
    if not body:
        return None

    # Check if body is base64-encoded
    if event.get("isBase64Encoded"):
        import base64

        body = base64.b64decode(body).decode("utf-8")

    if isinstance(body, str):
        return json.loads(body)

    return body


def _success_response(data: dict) -> dict:
    """Build a successful API Gateway response.

    Args:
        data: Response data to serialize

    Returns:
        API Gateway response dict
    """
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
        },
        "body": json.dumps(data),
    }


def _error_response(status_code: int, message: str) -> dict:
    """Build an error API Gateway response.

    Args:
        status_code: HTTP status code
        message: Error message

    Returns:
        API Gateway response dict
    """
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
        },
        "body": json.dumps({"error": message}),
    }


# For local testing
if __name__ == "__main__":
    # Set runtime mode
    os.environ["RUNTIME_MODE"] = "lambda"

    # Test event
    test_event = {
        "httpMethod": "POST",
        "path": "/invocations",
        "body": json.dumps({"prompt": "What is BERT?"}),
    }

    # Mock context
    class MockContext:
        aws_request_id = "test-request-id"

    result = handler(test_event, MockContext())
    print(json.dumps(result, indent=2))
