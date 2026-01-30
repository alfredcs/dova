"""
Logging Middleware for DOVA API.

Provides structured request/response logging with timing.
"""

import time
import uuid
from typing import Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured request logging.

    Logs:
    - Request method, path, and headers
    - Response status code and timing
    - Request ID for correlation
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with logging."""
        # Generate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Bind request context to logger
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        # Log request
        logger.info(
            "request_started",
            query_params=str(request.query_params),
            client_host=request.client.host if request.client else None,
        )

        # Process request
        start_time = time.time()
        try:
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000

            # Log response
            logger.info(
                "request_completed",
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.exception(
                "request_failed",
                error=str(e),
                duration_ms=round(duration_ms, 2),
            )
            raise

        finally:
            # Clear request context
            structlog.contextvars.unbind_contextvars("request_id", "method", "path")
