"""
Metrics Collection for DOVA.

Provides OpenTelemetry-based metrics collection for monitoring
agent performance, API latency, and resource usage.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point."""

    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)
    unit: str = ""


class MetricsCollector:
    """
    Collects and exports metrics for DOVA operations.

    In production, integrates with OpenTelemetry for CloudWatch/Prometheus export.
    In development, provides simple in-memory metrics storage.
    """

    def __init__(
        self,
        service_name: str = "dova",
        enable_otel: bool = False,
    ):
        self.service_name = service_name
        self.enable_otel = enable_otel
        self._metrics: list[MetricPoint] = []
        self._counters: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}
        self._meter = None

        if enable_otel:
            self._setup_otel()

    def _setup_otel(self) -> None:
        """Initialize OpenTelemetry metrics."""
        try:
            from opentelemetry import metrics
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.resources import Resource

            resource = Resource.create({"service.name": self.service_name})
            provider = MeterProvider(resource=resource)
            metrics.set_meter_provider(provider)
            self._meter = metrics.get_meter(self.service_name)
            logger.info("otel_metrics_initialized", service=self.service_name)
        except ImportError:
            logger.warning("otel_not_available", message="OpenTelemetry not installed")
            self.enable_otel = False

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Increment a counter metric.

        Args:
            name: Metric name
            value: Value to add (default 1)
            labels: Optional labels/tags
        """
        labels = labels or {}
        key = f"{name}:{':'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
        self._counters[key] = self._counters.get(key, 0) + value

        self._metrics.append(
            MetricPoint(name=name, value=value, labels=labels, unit="count")
        )

        if self.enable_otel and self._meter:
            counter = self._meter.create_counter(
                name,
                description=f"Counter for {name}",
            )
            counter.add(int(value), labels)

    def record(
        self,
        name: str,
        value: float,
        unit: str = "",
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Record a gauge/histogram metric.

        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement (e.g., "ms", "bytes")
            labels: Optional labels/tags
        """
        labels = labels or {}
        key = f"{name}:{':'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"

        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)

        self._metrics.append(
            MetricPoint(name=name, value=value, labels=labels, unit=unit)
        )

        if self.enable_otel and self._meter:
            histogram = self._meter.create_histogram(
                name,
                description=f"Histogram for {name}",
                unit=unit,
            )
            histogram.record(value, labels)

    @contextmanager
    def timer(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> Generator[None, None, None]:
        """
        Context manager for timing operations.

        Args:
            name: Metric name for the timing
            labels: Optional labels/tags

        Yields:
            None

        Example:
            with metrics.timer("api_request_latency", {"endpoint": "/search"}):
                await do_something()
        """
        start_time = time.time()
        try:
            yield
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            self.record(name, elapsed_ms, unit="ms", labels=labels)

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Get current value of a counter."""
        labels = labels or {}
        key = f"{name}:{':'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
        return self._counters.get(key, 0)

    def get_histogram_stats(
        self, name: str, labels: dict[str, str] | None = None
    ) -> dict[str, float]:
        """Get statistics for a histogram metric."""
        labels = labels or {}
        key = f"{name}:{':'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
        values = self._histograms.get(key, [])

        if not values:
            return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0, "p50": 0, "p99": 0}

        sorted_values = sorted(values)
        count = len(values)

        return {
            "count": count,
            "sum": sum(values),
            "avg": sum(values) / count,
            "min": min(values),
            "max": max(values),
            "p50": sorted_values[int(count * 0.50)] if count > 0 else 0,
            "p99": sorted_values[int(count * 0.99)] if count > 0 else 0,
        }

    def get_recent_metrics(self, limit: int = 100) -> list[MetricPoint]:
        """Get most recent metric points."""
        return self._metrics[-limit:]

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        self._metrics.clear()
        self._counters.clear()
        self._histograms.clear()


# Global metrics collector instance
_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def track_latency(
    name: str,
    labels: dict[str, str] | None = None,
) -> Any:
    """
    Decorator to track function execution latency.

    Args:
        name: Metric name
        labels: Optional labels/tags

    Returns:
        Decorated function
    """

    def decorator(func: Any) -> Any:
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            metrics = get_metrics_collector()
            with metrics.timer(name, labels=labels):
                return await func(*args, **kwargs)

        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            metrics = get_metrics_collector()
            with metrics.timer(name, labels=labels):
                return func(*args, **kwargs)

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Common metric names as constants
class MetricNames:
    """Standard metric names used throughout DOVA."""

    # Agent metrics
    AGENT_EXECUTION_LATENCY = "dova.agent.execution_latency_ms"
    AGENT_EXECUTION_COUNT = "dova.agent.execution_count"
    AGENT_ERROR_COUNT = "dova.agent.error_count"

    # MCP metrics
    MCP_CALL_LATENCY = "dova.mcp.call_latency_ms"
    MCP_CALL_COUNT = "dova.mcp.call_count"
    MCP_ERROR_COUNT = "dova.mcp.error_count"

    # LLM metrics
    LLM_CALL_LATENCY = "dova.llm.call_latency_ms"
    LLM_INPUT_TOKENS = "dova.llm.input_tokens"
    LLM_OUTPUT_TOKENS = "dova.llm.output_tokens"
    LLM_COST = "dova.llm.cost_usd"

    # API metrics
    API_REQUEST_LATENCY = "dova.api.request_latency_ms"
    API_REQUEST_COUNT = "dova.api.request_count"
    API_ERROR_COUNT = "dova.api.error_count"

    # Cache metrics
    CACHE_HIT_COUNT = "dova.cache.hit_count"
    CACHE_MISS_COUNT = "dova.cache.miss_count"
