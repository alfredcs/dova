"""
Job definitions for DOVA background processing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class JobStatus(Enum):
    """Job execution status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class JobPriority(Enum):
    """Job priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class JobType(Enum):
    """Types of background jobs."""

    ARXIV_POLL = "arxiv_poll"
    HF_POLL = "hf_poll"
    GITHUB_WEBHOOK = "github_webhook"
    CONTENT_PROCESS = "content_process"
    USER_MATCH = "user_match"
    DELIVERY = "delivery"
    SANDBOX_EXECUTE = "sandbox_execute"


@dataclass
class Job:
    """A background job to be processed."""

    type: JobType
    payload: dict[str, Any]
    id: UUID = field(default_factory=uuid4)
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retry_count: int = 0
    max_retries: int = 3
    error: str | None = None
    result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize job to dictionary."""
        return {
            "id": str(self.id),
            "type": self.type.value,
            "payload": self.payload,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "error": self.error,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Job":
        """Deserialize job from dictionary."""
        return cls(
            id=UUID(data["id"]),
            type=JobType(data["type"]),
            payload=data["payload"],
            priority=JobPriority(data["priority"]),
            status=JobStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=(
                datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            error=data.get("error"),
            result=data.get("result"),
        )
