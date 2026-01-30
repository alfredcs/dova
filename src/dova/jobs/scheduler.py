"""
Job scheduler for periodic background tasks.
"""

from typing import Callable

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from dova.jobs.jobs import Job, JobType
from dova.jobs.streams import JobQueue

logger = structlog.get_logger(__name__)


class DOVAScheduler:
    """
    Scheduler for periodic background jobs.

    Uses APScheduler to trigger jobs at configured intervals.
    """

    def __init__(
        self,
        job_queue: JobQueue,
        arxiv_poll_hours: float = 1.0,
        hf_poll_hours: float = 6.0,
    ):
        self.job_queue = job_queue
        self.arxiv_poll_hours = arxiv_poll_hours
        self.hf_poll_hours = hf_poll_hours
        self._scheduler = AsyncIOScheduler()
        self._logger = logger.bind(service="scheduler")

    async def start(self) -> None:
        """Start the scheduler with configured jobs."""
        await self.job_queue.initialize()

        # Schedule ArXiv polling
        self._scheduler.add_job(
            self._enqueue_arxiv_poll,
            trigger=IntervalTrigger(hours=self.arxiv_poll_hours),
            id="arxiv_poll",
            name="ArXiv Paper Poll",
            replace_existing=True,
        )

        # Schedule HuggingFace polling
        self._scheduler.add_job(
            self._enqueue_hf_poll,
            trigger=IntervalTrigger(hours=self.hf_poll_hours),
            id="hf_poll",
            name="HuggingFace Model Poll",
            replace_existing=True,
        )

        self._scheduler.start()
        self._logger.info(
            "scheduler_started",
            arxiv_hours=self.arxiv_poll_hours,
            hf_hours=self.hf_poll_hours,
        )

        # Run initial poll jobs immediately
        await self._enqueue_arxiv_poll()
        await self._enqueue_hf_poll()

    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        self._scheduler.shutdown(wait=True)
        self._logger.info("scheduler_stopped")

    async def _enqueue_arxiv_poll(self) -> None:
        """Enqueue an ArXiv polling job."""
        job = Job(
            type=JobType.ARXIV_POLL,
            payload={"categories": ["cs.AI", "cs.LG", "cs.CL"]},
        )
        await self.job_queue.enqueue(job)
        self._logger.debug("arxiv_poll_scheduled", job_id=str(job.id))

    async def _enqueue_hf_poll(self) -> None:
        """Enqueue a HuggingFace polling job."""
        job = Job(
            type=JobType.HF_POLL,
            payload={"tasks": ["text-generation", "image-classification"]},
        )
        await self.job_queue.enqueue(job)
        self._logger.debug("hf_poll_scheduled", job_id=str(job.id))

    def add_custom_job(
        self,
        job_id: str,
        func: Callable,
        hours: float = 1.0,
        name: str | None = None,
    ) -> None:
        """Add a custom scheduled job."""
        self._scheduler.add_job(
            func,
            trigger=IntervalTrigger(hours=hours),
            id=job_id,
            name=name or job_id,
            replace_existing=True,
        )
        self._logger.info("custom_job_added", job_id=job_id, hours=hours)
