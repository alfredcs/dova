"""
Retry Logic for DOVA.

Provides configurable retry mechanisms with exponential backoff,
jitter, and circuit breaker patterns for resilient operations.
"""

import asyncio
import random
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_range: tuple[float, float] = (0.5, 1.5)
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (Exception,)
    )
    non_retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (KeyboardInterrupt, SystemExit)
    )


def calculate_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """
    Calculate delay for a retry attempt with exponential backoff and jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    delay = config.base_delay * (config.exponential_base**attempt)
    delay = min(delay, config.max_delay)

    if config.jitter:
        jitter_factor = random.uniform(*config.jitter_range)
        delay *= jitter_factor

    return delay


async def retry_async(
    func: Callable[..., T],
    *args: Any,
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> T:
    """
    Execute an async function with retry logic.

    Args:
        func: Async function to execute
        *args: Positional arguments for the function
        config: Retry configuration (uses defaults if None)
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function

    Raises:
        The last exception if all retries are exhausted
    """
    config = config or RetryConfig()
    last_exception: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except config.non_retryable_exceptions:
            raise
        except config.retryable_exceptions as e:
            last_exception = e

            if attempt == config.max_retries:
                logger.error(
                    "retry_exhausted",
                    function=func.__name__,
                    attempts=attempt + 1,
                    error=str(e) or f"{type(e).__name__}()",
                )
                raise

            delay = calculate_delay(attempt, config)
            logger.warning(
                "retry_attempt",
                function=func.__name__,
                attempt=attempt + 1,
                max_retries=config.max_retries,
                delay=delay,
                error=str(e) or f"{type(e).__name__}()",
            )
            await asyncio.sleep(delay)

    # This should never be reached, but satisfies type checker
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected state in retry logic")


def with_retry(config: RetryConfig | None = None) -> Callable:
    """
    Decorator to add retry logic to an async function.

    Args:
        config: Retry configuration

    Returns:
        Decorated function with retry logic
    """
    config = config or RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry_async(func, *args, config=config, **kwargs)

        return wrapper

    return decorator


@dataclass
class CircuitBreakerState:
    """State for a circuit breaker."""

    failures: int = 0
    last_failure_time: float | None = None
    state: str = "closed"  # closed, open, half-open


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by stopping requests to failing services.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._states: dict[str, CircuitBreakerState] = {}
        self._half_open_calls: dict[str, int] = {}

    def _get_state(self, key: str) -> CircuitBreakerState:
        """Get or create circuit breaker state for a key."""
        if key not in self._states:
            self._states[key] = CircuitBreakerState()
        return self._states[key]

    async def call(
        self,
        key: str,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a function with circuit breaker protection.

        Args:
            key: Identifier for this circuit (e.g., service name)
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of the function

        Raises:
            CircuitBreakerOpen: If the circuit is open
            The original exception if the call fails
        """
        import time

        state = self._get_state(key)
        now = time.time()

        # Check if circuit is open
        if state.state == "open":
            if state.last_failure_time and (
                now - state.last_failure_time > self.recovery_timeout
            ):
                # Transition to half-open
                state.state = "half-open"
                self._half_open_calls[key] = 0
                logger.info("circuit_breaker_half_open", key=key)
            else:
                raise CircuitBreakerOpen(f"Circuit breaker is open for {key}")

        # Check half-open call limit
        if state.state == "half-open":
            if self._half_open_calls.get(key, 0) >= self.half_open_max_calls:
                raise CircuitBreakerOpen(f"Circuit breaker half-open limit reached for {key}")
            self._half_open_calls[key] = self._half_open_calls.get(key, 0) + 1

        try:
            result = await func(*args, **kwargs)

            # Success: reset state
            if state.state == "half-open":
                state.state = "closed"
                state.failures = 0
                logger.info("circuit_breaker_closed", key=key)
            elif state.state == "closed":
                state.failures = 0

            return result

        except Exception as e:
            state.failures += 1
            state.last_failure_time = now

            if state.state == "half-open" or state.failures >= self.failure_threshold:
                state.state = "open"
                logger.warning(
                    "circuit_breaker_open",
                    key=key,
                    failures=state.failures,
                    error=str(e),
                )

            raise


class CircuitBreakerOpen(Exception):
    """Raised when attempting to call through an open circuit breaker."""

    pass
