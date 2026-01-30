"""
Redis Streams wrapper for job queue.
"""

import json
from typing import Any

import structlog
from redis.asyncio import Redis

from dova.jobs.jobs import Job, JobPriority

logger = structlog.get_logger(__name__)


class JobQueue:
    """
    Redis Streams-based job queue.

    Uses Redis Streams for reliable job delivery with consumer groups.
    """

    def __init__(
        self,
        redis: Redis,
        stream_name: str = "dova:jobs",
        group_name: str = "dova-workers",
    ):
        self.redis = redis
        self.stream_name = stream_name
        self.group_name = group_name
        self._logger = logger.bind(service="job_queue")

    async def initialize(self) -> None:
        """Initialize the consumer group if it doesn't exist."""
        try:
            await self.redis.xgroup_create(
                self.stream_name,
                self.group_name,
                id="0",
                mkstream=True,
            )
            self._logger.info(
                "consumer_group_created",
                stream=self.stream_name,
                group=self.group_name,
            )
        except Exception as e:
            if "BUSYGROUP" in str(e):
                self._logger.debug("consumer_group_exists", group=self.group_name)
            else:
                raise

    async def enqueue(self, job: Job) -> str:
        """
        Add a job to the queue.

        Returns:
            Message ID from Redis Stream
        """
        job_data = json.dumps(job.to_dict())
        message_id = await self.redis.xadd(
            self.stream_name,
            {"job": job_data, "priority": str(job.priority.value)},
        )
        self._logger.debug(
            "job_enqueued",
            job_id=str(job.id),
            job_type=job.type.value,
            message_id=message_id,
        )
        return message_id

    async def dequeue(
        self,
        consumer_name: str,
        count: int = 1,
        block_ms: int = 5000,
    ) -> list[tuple[str, Job]]:
        """
        Read jobs from the queue.

        Args:
            consumer_name: Unique consumer identifier
            count: Max jobs to fetch
            block_ms: How long to block waiting for jobs

        Returns:
            List of (message_id, Job) tuples
        """
        try:
            messages = await self.redis.xreadgroup(
                self.group_name,
                consumer_name,
                {self.stream_name: ">"},
                count=count,
                block=block_ms,
            )
        except Exception as e:
            self._logger.error("dequeue_error", error=str(e))
            return []

        if not messages:
            return []

        jobs: list[tuple[str, Job]] = []
        for _, stream_messages in messages:
            for message_id, data in stream_messages:
                try:
                    job_data = json.loads(data[b"job"].decode())
                    job = Job.from_dict(job_data)
                    jobs.append((message_id.decode(), job))
                except (json.JSONDecodeError, KeyError) as e:
                    self._logger.error("job_parse_error", message_id=message_id, error=str(e))
                    await self.ack(message_id.decode())

        return jobs

    async def ack(self, message_id: str) -> None:
        """Acknowledge job completion."""
        await self.redis.xack(self.stream_name, self.group_name, message_id)
        self._logger.debug("job_acknowledged", message_id=message_id)

    async def get_pending_count(self) -> int:
        """Get count of pending jobs in the stream."""
        info = await self.redis.xinfo_groups(self.stream_name)
        for group in info:
            if group.get("name", b"").decode() == self.group_name:
                return group.get("pending", 0)
        return 0

    async def get_stream_length(self) -> int:
        """Get total messages in the stream."""
        return await self.redis.xlen(self.stream_name)

    async def claim_stale_jobs(
        self,
        consumer_name: str,
        min_idle_time_ms: int = 60000,
        count: int = 10,
    ) -> list[tuple[str, Job]]:
        """
        Claim jobs that have been pending too long (stuck consumers).

        Args:
            consumer_name: Consumer claiming the jobs
            min_idle_time_ms: Minimum idle time before claiming
            count: Max jobs to claim

        Returns:
            List of claimed (message_id, Job) tuples
        """
        try:
            pending = await self.redis.xpending_range(
                self.stream_name,
                self.group_name,
                min="-",
                max="+",
                count=count,
            )
        except Exception as e:
            self._logger.error("pending_range_error", error=str(e))
            return []

        stale_ids = [
            entry["message_id"].decode()
            for entry in pending
            if entry["time_since_delivered"] >= min_idle_time_ms
        ]

        if not stale_ids:
            return []

        claimed = await self.redis.xclaim(
            self.stream_name,
            self.group_name,
            consumer_name,
            min_idle_time_ms,
            stale_ids,
        )

        jobs: list[tuple[str, Job]] = []
        for message_id, data in claimed:
            try:
                job_data = json.loads(data[b"job"].decode())
                job = Job.from_dict(job_data)
                jobs.append((message_id.decode(), job))
                self._logger.info(
                    "job_claimed",
                    message_id=message_id.decode(),
                    job_id=str(job.id),
                )
            except (json.JSONDecodeError, KeyError) as e:
                self._logger.error("claimed_job_parse_error", error=str(e))

        return jobs
