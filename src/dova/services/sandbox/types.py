"""
Sandbox type definitions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class SandboxTier(Enum):
    """Sandbox resource tiers."""

    CPU_BASIC = "cpu_basic"  # 0.5 vCPU, 512MB, 60s
    CPU_STANDARD = "cpu_standard"  # 2 vCPU, 4GB, 300s
    GPU_SPOT = "gpu_spot"  # T4 GPU, 16GB, 600s
    GPU_PREMIUM = "gpu_premium"  # A10 GPU, 32GB, 1800s


class ExecutionStatus(Enum):
    """Execution job status."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class TierConfig:
    """Configuration for a sandbox tier."""

    tier: SandboxTier
    cpu_limit: float  # vCPUs
    memory_mb: int  # Memory in MB
    timeout_seconds: int  # Max execution time
    gpu_enabled: bool = False
    gpu_type: str | None = None
    cost_per_second: float = 0.0  # For quota tracking

    @classmethod
    def get_config(cls, tier: SandboxTier) -> "TierConfig":
        """Get configuration for a tier."""
        configs = {
            SandboxTier.CPU_BASIC: cls(
                tier=SandboxTier.CPU_BASIC,
                cpu_limit=0.5,
                memory_mb=512,
                timeout_seconds=60,
            ),
            SandboxTier.CPU_STANDARD: cls(
                tier=SandboxTier.CPU_STANDARD,
                cpu_limit=2.0,
                memory_mb=4096,
                timeout_seconds=300,
            ),
            SandboxTier.GPU_SPOT: cls(
                tier=SandboxTier.GPU_SPOT,
                cpu_limit=4.0,
                memory_mb=16384,
                timeout_seconds=600,
                gpu_enabled=True,
                gpu_type="nvidia-t4",
                cost_per_second=0.001,
            ),
            SandboxTier.GPU_PREMIUM: cls(
                tier=SandboxTier.GPU_PREMIUM,
                cpu_limit=8.0,
                memory_mb=32768,
                timeout_seconds=1800,
                gpu_enabled=True,
                gpu_type="nvidia-a10g",
                cost_per_second=0.005,
            ),
        }
        return configs[tier]


@dataclass
class SandboxQuota:
    """User's sandbox execution quota."""

    user_id: str
    daily_cpu_seconds: int = 3600  # 1 hour CPU time per day
    daily_gpu_seconds: int = 600  # 10 minutes GPU time per day
    used_cpu_seconds: float = 0.0
    used_gpu_seconds: float = 0.0
    quota_reset_at: datetime = field(default_factory=datetime.utcnow)
    max_concurrent: int = 3

    def can_execute(self, tier: SandboxTier, estimated_seconds: int) -> bool:
        """Check if quota allows execution."""
        config = TierConfig.get_config(tier)
        if config.gpu_enabled:
            return self.used_gpu_seconds + estimated_seconds <= self.daily_gpu_seconds
        return self.used_cpu_seconds + estimated_seconds <= self.daily_cpu_seconds

    def record_usage(self, tier: SandboxTier, seconds: float) -> None:
        """Record resource usage."""
        config = TierConfig.get_config(tier)
        if config.gpu_enabled:
            self.used_gpu_seconds += seconds
        else:
            self.used_cpu_seconds += seconds


@dataclass
class SandboxJob:
    """A sandbox execution job."""

    id: UUID = field(default_factory=uuid4)
    user_id: str = ""
    code: str = ""
    language: str = "python"
    tier: SandboxTier = SandboxTier.CPU_BASIC
    status: ExecutionStatus = ExecutionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    timeout_seconds: int = 60
    dependencies: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    input_data: dict[str, Any] | None = None
    output: str | None = None
    error: str | None = None
    exit_code: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "code": self.code,
            "language": self.language,
            "tier": self.tier.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "timeout_seconds": self.timeout_seconds,
            "dependencies": self.dependencies,
            "environment": self.environment,
            "input_data": self.input_data,
            "output": self.output,
            "error": self.error,
            "exit_code": self.exit_code,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SandboxJob":
        """Deserialize from dictionary."""
        return cls(
            id=UUID(data["id"]),
            user_id=data["user_id"],
            code=data["code"],
            language=data.get("language", "python"),
            tier=SandboxTier(data["tier"]),
            status=ExecutionStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=(
                datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
            ),
            timeout_seconds=data.get("timeout_seconds", 60),
            dependencies=data.get("dependencies", []),
            environment=data.get("environment", {}),
            input_data=data.get("input_data"),
            output=data.get("output"),
            error=data.get("error"),
            exit_code=data.get("exit_code"),
            metadata=data.get("metadata", {}),
        )
