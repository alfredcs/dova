"""
Proactive heartbeat task system.

Provides cron-based scheduling for background maintenance tasks
like health checks, cleanups, and periodic refreshes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from dova.jobs.streams import JobQueue

logger = structlog.get_logger(__name__)

# Type for async handler functions
HandlerFunc = Callable[[], Coroutine[Any, Any, None]]


@dataclass
class HeartbeatTask:
    """Definition of a heartbeat task."""

    name: str
    cron_schedule: str  # Cron expression (e.g., "*/15 * * * *")
    handler: str  # Handler name to call
    enabled: bool = True
    last_run: datetime | None = None
    run_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "cron_schedule": self.cron_schedule,
            "handler": self.handler,
            "enabled": self.enabled,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "run_count": self.run_count,
            "metadata": self.metadata,
        }


# Default heartbeat tasks
DEFAULT_TASKS: list[HeartbeatTask] = [
    HeartbeatTask(
        name="subscription_monitor",
        cron_schedule="*/15 * * * *",  # Every 15 minutes
        handler="check_subscriptions",
        metadata={"description": "Monitor subscription feeds for updates"},
    ),
    HeartbeatTask(
        name="recommendation_refresh",
        cron_schedule="0 */4 * * *",  # Every 4 hours
        handler="refresh_recommendations",
        metadata={"description": "Refresh user recommendations"},
    ),
    HeartbeatTask(
        name="mcp_health_check",
        cron_schedule="*/5 * * * *",  # Every 5 minutes
        handler="check_mcp_health",
        metadata={"description": "Check MCP server health"},
    ),
    HeartbeatTask(
        name="session_cleanup",
        cron_schedule="0 3 * * *",  # Daily at 3 AM
        handler="cleanup_sessions",
        metadata={"description": "Clean up expired sessions"},
    ),
    HeartbeatTask(
        name="mcp_repo_update",
        cron_schedule="0 4 * * 0",  # Weekly on Sunday at 4 AM
        handler="update_mcp_repos",
        metadata={"description": "Git pull updates for MCP server repos"},
    ),
]


class HeartbeatProcessor:
    """
    Manages proactive heartbeat tasks.

    Uses APScheduler for cron-based task scheduling with support
    for custom handlers and job queue integration.
    """

    def __init__(
        self,
        job_queue: JobQueue | None = None,
        auto_register_defaults: bool = True,
    ):
        self.job_queue = job_queue
        self._scheduler = AsyncIOScheduler()
        self._tasks: dict[str, HeartbeatTask] = {}
        self._handlers: dict[str, HandlerFunc] = {}
        self._running = False
        self._logger = logger.bind(service="heartbeat")

        # Register built-in handlers
        self._register_builtin_handlers()

        # Register default tasks if requested
        if auto_register_defaults:
            for task in DEFAULT_TASKS:
                self._tasks[task.name] = task

    def register_handler(self, name: str, handler: HandlerFunc) -> None:
        """
        Register a handler function.

        Args:
            name: Handler name (matches HeartbeatTask.handler)
            handler: Async function to call
        """
        self._handlers[name] = handler
        self._logger.debug("handler_registered", name=name)

    def register_task(self, task: HeartbeatTask) -> None:
        """
        Register a heartbeat task.

        Args:
            task: Task definition
        """
        self._tasks[task.name] = task
        self._logger.info(
            "task_registered",
            name=task.name,
            schedule=task.cron_schedule,
        )

        # If already running, add to scheduler immediately
        if self._running:
            self._add_task_to_scheduler(task)

    def unregister_task(self, name: str) -> bool:
        """
        Unregister a heartbeat task.

        Args:
            name: Task name

        Returns:
            True if task was removed
        """
        if name not in self._tasks:
            return False

        del self._tasks[name]

        if self._running:
            try:
                self._scheduler.remove_job(f"heartbeat:{name}")
            except Exception:
                pass

        self._logger.info("task_unregistered", name=name)
        return True

    def enable_task(self, name: str) -> bool:
        """Enable a task."""
        if name not in self._tasks:
            return False
        self._tasks[name].enabled = True
        return True

    def disable_task(self, name: str) -> bool:
        """Disable a task."""
        if name not in self._tasks:
            return False
        self._tasks[name].enabled = False
        return True

    def get_task(self, name: str) -> HeartbeatTask | None:
        """Get a task by name."""
        return self._tasks.get(name)

    def list_tasks(self) -> list[HeartbeatTask]:
        """List all registered tasks."""
        return list(self._tasks.values())

    async def start(self) -> None:
        """Start the heartbeat processor."""
        if self._running:
            self._logger.warning("heartbeat_already_running")
            return

        # Add all enabled tasks to scheduler
        for task in self._tasks.values():
            if task.enabled:
                self._add_task_to_scheduler(task)

        self._scheduler.start()
        self._running = True

        self._logger.info(
            "heartbeat_started",
            tasks=len(self._tasks),
            enabled=sum(1 for t in self._tasks.values() if t.enabled),
        )

    async def stop(self) -> None:
        """Stop the heartbeat processor."""
        if not self._running:
            return

        self._scheduler.shutdown(wait=True)
        self._running = False
        self._logger.info("heartbeat_stopped")

    async def run_task_now(self, name: str) -> bool:
        """
        Run a task immediately.

        Args:
            name: Task name

        Returns:
            True if task was executed
        """
        task = self._tasks.get(name)
        if task is None:
            return False

        await self._execute_task(task)
        return True

    def _add_task_to_scheduler(self, task: HeartbeatTask) -> None:
        """Add a task to the APScheduler."""
        try:
            trigger = CronTrigger.from_crontab(task.cron_schedule)
            self._scheduler.add_job(
                self._execute_task,
                trigger=trigger,
                args=[task],
                id=f"heartbeat:{task.name}",
                name=task.name,
                replace_existing=True,
            )
            self._logger.debug(
                "task_scheduled",
                name=task.name,
                schedule=task.cron_schedule,
            )
        except Exception as e:
            self._logger.error(
                "task_schedule_failed",
                name=task.name,
                error=str(e),
            )

    async def _execute_task(self, task: HeartbeatTask) -> None:
        """Execute a heartbeat task."""
        if not task.enabled:
            return

        handler = self._handlers.get(task.handler)
        if handler is None:
            self._logger.warning(
                "handler_not_found",
                task=task.name,
                handler=task.handler,
            )
            return

        try:
            self._logger.debug("task_executing", name=task.name)
            await handler()
            task.last_run = datetime.utcnow()
            task.run_count += 1
            self._logger.info(
                "task_completed",
                name=task.name,
                run_count=task.run_count,
            )
        except Exception as e:
            self._logger.error(
                "task_failed",
                name=task.name,
                error=str(e),
            )

    def _register_builtin_handlers(self) -> None:
        """Register built-in handler implementations."""

        async def check_subscriptions() -> None:
            """Check subscription feeds for updates."""
            self._logger.debug("checking_subscriptions")
            # In production, this would check feed sources

        async def refresh_recommendations() -> None:
            """Refresh user recommendations."""
            self._logger.debug("refreshing_recommendations")
            # In production, this would update recommendation cache

        async def check_mcp_health() -> None:
            """Check MCP server health."""
            self._logger.debug("checking_mcp_health")
            # In production, this would ping MCP servers

        async def cleanup_sessions() -> None:
            """Clean up expired sessions."""
            self._logger.debug("cleaning_up_sessions")
            # In production, this would remove expired sessions

        async def update_mcp_repos() -> None:
            """Git pull updates for MCP server repos."""
            self._logger.info("updating_mcp_repos")
            try:
                from dova.services.mcp_repo_manager import update_mcp_repos as do_update

                results = await do_update()
                self._logger.info(
                    "mcp_repos_updated",
                    results=results,
                )
            except Exception as e:
                self._logger.error("mcp_repo_update_failed", error=str(e))

        self._handlers["check_subscriptions"] = check_subscriptions
        self._handlers["refresh_recommendations"] = refresh_recommendations
        self._handlers["check_mcp_health"] = check_mcp_health
        self._handlers["cleanup_sessions"] = cleanup_sessions
        self._handlers["update_mcp_repos"] = update_mcp_repos
