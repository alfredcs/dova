"""Concurrency helpers for DOVA entry points.

Three capabilities:

1. ``configure_default_executor`` — enlarge the asyncio event loop's default
   ``ThreadPoolExecutor``. DOVA's Bedrock provider calls
   ``boto3.invoke_model`` via ``run_in_executor(None, ...)`` because botocore
   is synchronous. Python's default is ``min(32, cpu_count + 4)`` ≈ 8 on a
   typical host. With ~9 concurrent MCP requests × 3-6 LLM calls each, the
   default pool saturates and every call blocks on the queue.

2. ``request_tracker`` / ``track_request`` — a process-wide counter of
   in-flight requests plus a peak high-water mark. Use as an async context
   manager at MCP tool entry points so we have real data on actual
   concurrency rather than hypothesized numbers.

3. ``start_saturation_logger`` — background task that periodically logs the
   default executor's pending-queue depth and active worker count while any
   request is in flight. Lets us see whether the ``max_workers`` value is
   actually large enough.

Call ``configure_default_executor`` once at process startup (from
``dova serve`` and ``dova mcp serve``). ``start_saturation_logger`` is
optional — wire it in after the default executor is configured.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator

import structlog

logger = structlog.get_logger(__name__)

_configured: bool = False


# ---------------------------------------------------------------------------
# Default ThreadPoolExecutor sizing
# ---------------------------------------------------------------------------

def configure_default_executor(max_workers: int | None = None) -> int:
    """Replace the event loop's default executor with a larger one.

    Args:
        max_workers: Explicit worker count. When omitted, reads
            ``DOVA_EXECUTOR_WORKERS`` from the environment, falling back to
            ``64`` — enough headroom for ~10 concurrent requests each making
            ~6 synchronous boto3 calls without queuing.

    Returns:
        The configured worker count.

    Idempotent: a second call is a no-op and returns the original count.
    """
    global _configured
    if _configured:
        return _current_max_workers()

    if max_workers is None:
        env = os.environ.get("DOVA_EXECUTOR_WORKERS")
        max_workers = int(env) if env and env.isdigit() else 64

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="dova-blocking",
    )
    loop.set_default_executor(executor)
    _configured = True
    logger.info("default_executor_configured", max_workers=max_workers)
    return max_workers


def _current_max_workers() -> int:
    """Best-effort read of the current default executor's worker count."""
    loop = asyncio.get_event_loop()
    executor = getattr(loop, "_default_executor", None)
    if isinstance(executor, ThreadPoolExecutor):
        return executor._max_workers
    return 0


# ---------------------------------------------------------------------------
# In-flight request gauge (#8)
# ---------------------------------------------------------------------------

class _RequestTracker:
    """Process-wide counter of in-flight + peak concurrent requests.

    Intentionally simple: a single integer incremented on entry and
    decremented on exit, plus a peak high-water mark that resets only on
    process restart. No locks — the GIL makes ``int += 1`` safe enough for
    a monotonic counter like this; the peak comparison is a best-effort
    reading, not a strict invariant.
    """

    def __init__(self) -> None:
        self.in_flight: int = 0
        self.peak: int = 0
        self.total: int = 0

    def enter(self) -> int:
        self.in_flight += 1
        self.total += 1
        if self.in_flight > self.peak:
            self.peak = self.in_flight
        return self.in_flight

    def exit(self) -> int:
        self.in_flight = max(0, self.in_flight - 1)
        return self.in_flight


_request_tracker = _RequestTracker()


def request_tracker() -> _RequestTracker:
    """Return the process-wide request tracker singleton."""
    return _request_tracker


def tracked(label: str):
    """Decorator that wraps an async function in ``track_request(label)``.

    Example::

        @tracked("dova_research")
        async def dova_research(query: str) -> str:
            ...
    """
    from functools import wraps

    def decorator(fn):
        @wraps(fn)
        async def inner(*args, **kwargs):
            async with track_request(label):
                return await fn(*args, **kwargs)
        return inner

    return decorator


@contextlib.asynccontextmanager
async def track_request(label: str) -> AsyncIterator[int]:
    """Async context manager that increments/decrements the in-flight counter.

    Logs the current in-flight count on entry and the request's share of the
    peak on exit. Use at MCP tool entry points so every request contributes
    to the counter:

        async with track_request("dova_research"):
            ...

    Yields the in-flight count at the moment of entry.
    """
    t = _request_tracker
    current = t.enter()
    logger.info(
        "request_started",
        tool=label,
        in_flight=current,
        peak=t.peak,
    )
    try:
        yield current
    finally:
        remaining = t.exit()
        logger.info(
            "request_finished",
            tool=label,
            in_flight=remaining,
            peak=t.peak,
            total=t.total,
        )


# ---------------------------------------------------------------------------
# Executor saturation logger (#9)
# ---------------------------------------------------------------------------

_saturation_task: asyncio.Task | None = None


def start_saturation_logger(interval_s: float = 5.0) -> None:
    """Start a background task that logs default-executor saturation.

    Emits a structured log line every ``interval_s`` seconds whenever at
    least one request is in flight. Silent when idle so dev logs don't fill
    with noise. Idempotent — a second call is a no-op.

    Fields logged:
        workers_max: configured max_workers on the executor
        workers_busy: best-effort count of threads currently running a task
        queue_depth: pending submissions waiting for a free thread
        in_flight_requests: from the request tracker
        peak_requests: peak high-water mark since process start
    """
    global _saturation_task
    if _saturation_task is not None and not _saturation_task.done():
        return

    async def _loop() -> None:
        while True:
            try:
                await asyncio.sleep(interval_s)
                t = _request_tracker
                if t.in_flight == 0:
                    continue
                loop = asyncio.get_event_loop()
                executor = getattr(loop, "_default_executor", None)
                if not isinstance(executor, ThreadPoolExecutor):
                    continue
                # _work_queue / _threads are implementation details but
                # stable across cpython 3.11+; tolerate their absence.
                try:
                    queue_depth = executor._work_queue.qsize()  # type: ignore[attr-defined]
                except Exception:
                    queue_depth = -1
                try:
                    workers_busy = sum(
                        1 for th in executor._threads if th.is_alive()  # type: ignore[attr-defined]
                    )
                except Exception:
                    workers_busy = -1
                logger.info(
                    "executor_saturation",
                    workers_max=executor._max_workers,
                    workers_busy=workers_busy,
                    queue_depth=queue_depth,
                    in_flight_requests=t.in_flight,
                    peak_requests=t.peak,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("saturation_logger_error", error=str(exc))

    _saturation_task = asyncio.create_task(_loop(), name="dova-saturation-logger")
    logger.info("saturation_logger_started", interval_s=interval_s)


async def stop_saturation_logger() -> None:
    """Cancel the background saturation logger. Safe to call multiple times."""
    global _saturation_task
    task = _saturation_task
    _saturation_task = None
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
