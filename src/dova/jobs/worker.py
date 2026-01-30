"""
Background job worker for processing queue items.
"""

import asyncio
import signal
from datetime import datetime
from typing import Any, Callable, Coroutine
from uuid import uuid4

import structlog
from redis.asyncio import Redis

from dova.jobs.jobs import Job, JobStatus, JobType
from dova.jobs.streams import JobQueue

logger = structlog.get_logger(__name__)

JobHandler = Callable[[Job], Coroutine[Any, Any, dict[str, Any] | None]]


class JobWorker:
    """
    Worker that processes jobs from the queue.

    Supports multiple concurrent handlers and graceful shutdown.
    """

    def __init__(
        self,
        job_queue: JobQueue,
        worker_id: str | None = None,
        concurrency: int = 5,
    ):
        self.job_queue = job_queue
        self.worker_id = worker_id or f"worker-{uuid4().hex[:8]}"
        self.concurrency = concurrency
        self._handlers: dict[JobType, JobHandler] = {}
        self._running = False
        self._tasks: set[asyncio.Task] = set()
        self._logger = logger.bind(service="worker", worker_id=self.worker_id)

    def register_handler(self, job_type: JobType, handler: JobHandler) -> None:
        """Register a handler for a job type."""
        self._handlers[job_type] = handler
        self._logger.debug("handler_registered", job_type=job_type.value)

    async def start(self) -> None:
        """Start the worker processing loop."""
        self._running = True
        self._logger.info("worker_starting", concurrency=self.concurrency)

        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

        await self.job_queue.initialize()

        while self._running:
            # Claim any stale jobs first
            stale_jobs = await self.job_queue.claim_stale_jobs(self.worker_id)
            for message_id, job in stale_jobs:
                self._schedule_job(message_id, job)

            # Check if we have capacity
            if len(self._tasks) >= self.concurrency:
                await asyncio.sleep(0.1)
                continue

            # Fetch new jobs
            jobs = await self.job_queue.dequeue(
                self.worker_id,
                count=self.concurrency - len(self._tasks),
                block_ms=1000,
            )

            for message_id, job in jobs:
                self._schedule_job(message_id, job)

        # Wait for remaining tasks
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        self._logger.info("worker_stopped")

    async def stop(self) -> None:
        """Signal the worker to stop."""
        self._logger.info("worker_stopping")
        self._running = False

    def _schedule_job(self, message_id: str, job: Job) -> None:
        """Schedule a job for processing."""
        task = asyncio.create_task(self._process_job(message_id, job))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _process_job(self, message_id: str, job: Job) -> None:
        """Process a single job."""
        handler = self._handlers.get(job.type)
        if not handler:
            self._logger.warning("no_handler", job_type=job.type.value, job_id=str(job.id))
            await self.job_queue.ack(message_id)
            return

        job.status = JobStatus.IN_PROGRESS
        job.started_at = datetime.utcnow()

        self._logger.info(
            "job_processing",
            job_id=str(job.id),
            job_type=job.type.value,
            retry_count=job.retry_count,
        )

        try:
            result = await handler(job)
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.result = result

            self._logger.info(
                "job_completed",
                job_id=str(job.id),
                job_type=job.type.value,
                duration_ms=(job.completed_at - job.started_at).total_seconds() * 1000,
            )

        except Exception as e:
            job.error = str(e)
            job.retry_count += 1

            if job.retry_count < job.max_retries:
                job.status = JobStatus.RETRYING
                self._logger.warning(
                    "job_retrying",
                    job_id=str(job.id),
                    retry_count=job.retry_count,
                    error=str(e),
                )
                # Re-enqueue for retry
                await self.job_queue.enqueue(job)
            else:
                job.status = JobStatus.FAILED
                self._logger.error(
                    "job_failed",
                    job_id=str(job.id),
                    job_type=job.type.value,
                    error=str(e),
                )

        await self.job_queue.ack(message_id)


async def run_worker(redis_url: str) -> None:
    """Run the worker as a standalone process."""
    from dova.config.settings import get_settings
    from dova.utils.logging import configure_logging

    settings = get_settings()
    configure_logging(
        log_level=settings.log_level,
        json_logs=settings.is_production,
        service_name="dova-worker",
    )

    redis = Redis.from_url(redis_url)
    job_queue = JobQueue(redis)
    worker = JobWorker(job_queue, concurrency=settings.jobs.worker_concurrency)

    # Import and register handlers
    from dova.services.recommendation.monitors import (
        handle_arxiv_poll,
        handle_hf_poll,
    )

    worker.register_handler(JobType.ARXIV_POLL, handle_arxiv_poll)
    worker.register_handler(JobType.HF_POLL, handle_hf_poll)

    await worker.start()
    await redis.close()


if __name__ == "__main__":
    import sys

    from dova.config.settings import get_settings

    settings = get_settings()
    asyncio.run(run_worker(settings.redis.url))
