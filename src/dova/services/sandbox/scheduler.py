"""
Sandbox scheduler for tier detection and job routing.
"""

import re
from typing import Any

import structlog

from dova.services.sandbox.types import SandboxJob, SandboxTier, TierConfig

logger = structlog.get_logger(__name__)

# Patterns indicating GPU-heavy workloads
GPU_PATTERNS = [
    r"\bimport\s+torch\b",
    r"\bfrom\s+torch\b",
    r"\bimport\s+tensorflow\b",
    r"\bfrom\s+tensorflow\b",
    r"\bimport\s+jax\b",
    r"\bfrom\s+jax\b",
    r"\b\.cuda\(\)",
    r"\b\.to\(['\"]cuda['\"]\)",
    r"\btorch\.nn\b",
    r"\btf\.keras\b",
    r"\bmodel\.fit\b",
    r"\bmodel\.train\b",
]

# Patterns indicating heavy CPU workloads
HEAVY_CPU_PATTERNS = [
    r"\bimport\s+numpy\b",
    r"\bimport\s+scipy\b",
    r"\bimport\s+pandas\b",
    r"\bimport\s+sklearn\b",
    r"\bfrom\s+sklearn\b",
    r"\bimport\s+cv2\b",
    r"\bimport\s+PIL\b",
    r"\bnp\.dot\b",
    r"\bnp\.matmul\b",
    r"\bparallel\b",
    r"\bmultiprocessing\b",
]

# Dependencies that suggest GPU usage
GPU_DEPENDENCIES = [
    "torch",
    "torchvision",
    "tensorflow",
    "tensorflow-gpu",
    "jax",
    "jaxlib",
    "cupy",
    "pycuda",
    "transformers",
    "diffusers",
]

# Dependencies that suggest heavy CPU usage
HEAVY_CPU_DEPENDENCIES = [
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "opencv-python",
    "pillow",
    "xgboost",
    "lightgbm",
]


class SandboxScheduler:
    """
    Schedules sandbox jobs with automatic tier detection.

    Analyzes code to determine appropriate resource tier.
    """

    def __init__(self, default_tier: SandboxTier = SandboxTier.CPU_BASIC):
        self.default_tier = default_tier
        self._gpu_patterns = [re.compile(p, re.IGNORECASE) for p in GPU_PATTERNS]
        self._cpu_patterns = [re.compile(p, re.IGNORECASE) for p in HEAVY_CPU_PATTERNS]
        self._logger = logger.bind(service="sandbox_scheduler")

    def infer_tier(
        self,
        code: str,
        dependencies: list[str] | None = None,
        explicit_tier: SandboxTier | None = None,
    ) -> SandboxTier:
        """
        Infer appropriate tier based on code analysis.

        Args:
            code: Source code to analyze
            dependencies: Explicit list of dependencies
            explicit_tier: User-specified tier (takes precedence)

        Returns:
            Recommended SandboxTier
        """
        if explicit_tier:
            return explicit_tier

        dependencies = dependencies or []
        deps_lower = [d.lower() for d in dependencies]

        # Check for GPU indicators
        gpu_score = 0

        # Check code patterns
        for pattern in self._gpu_patterns:
            if pattern.search(code):
                gpu_score += 2

        # Check dependencies
        for dep in GPU_DEPENDENCIES:
            if dep.lower() in deps_lower or any(dep.lower() in d for d in deps_lower):
                gpu_score += 3

        if gpu_score >= 3:
            self._logger.debug("tier_inferred", tier="gpu", score=gpu_score)
            return SandboxTier.GPU_SPOT

        # Check for heavy CPU indicators
        cpu_score = 0

        for pattern in self._cpu_patterns:
            if pattern.search(code):
                cpu_score += 1

        for dep in HEAVY_CPU_DEPENDENCIES:
            if dep.lower() in deps_lower or any(dep.lower() in d for d in deps_lower):
                cpu_score += 2

        if cpu_score >= 4:
            self._logger.debug("tier_inferred", tier="cpu_standard", score=cpu_score)
            return SandboxTier.CPU_STANDARD

        # Default to basic
        return self.default_tier

    def estimate_timeout(
        self,
        tier: SandboxTier,
        code_length: int,
        dependencies: list[str] | None = None,
    ) -> int:
        """
        Estimate appropriate timeout based on tier and code.

        Returns:
            Timeout in seconds
        """
        config = TierConfig.get_config(tier)

        # Base timeout from tier config
        base_timeout = config.timeout_seconds

        # Adjust based on code length (rough heuristic)
        if code_length > 5000:
            base_timeout = min(base_timeout * 2, config.timeout_seconds)
        elif code_length > 1000:
            base_timeout = min(int(base_timeout * 1.5), config.timeout_seconds)

        # Adjust for heavy dependencies
        dependencies = dependencies or []
        deps_lower = [d.lower() for d in dependencies]

        heavy_deps = GPU_DEPENDENCIES + HEAVY_CPU_DEPENDENCIES
        heavy_count = sum(1 for d in heavy_deps if d.lower() in deps_lower)

        if heavy_count >= 3:
            base_timeout = min(int(base_timeout * 1.5), config.timeout_seconds)

        return base_timeout

    def create_job(
        self,
        user_id: str,
        code: str,
        language: str = "python",
        dependencies: list[str] | None = None,
        environment: dict[str, str] | None = None,
        explicit_tier: SandboxTier | None = None,
        input_data: dict[str, Any] | None = None,
    ) -> SandboxJob:
        """
        Create a sandbox job with inferred configuration.

        Args:
            user_id: User requesting execution
            code: Source code to execute
            language: Programming language
            dependencies: Required packages
            environment: Environment variables
            explicit_tier: Override auto-detected tier
            input_data: Input data for the code

        Returns:
            Configured SandboxJob
        """
        dependencies = dependencies or []

        # Infer tier
        tier = self.infer_tier(code, dependencies, explicit_tier)

        # Estimate timeout
        timeout = self.estimate_timeout(tier, len(code), dependencies)

        job = SandboxJob(
            user_id=user_id,
            code=code,
            language=language,
            tier=tier,
            timeout_seconds=timeout,
            dependencies=dependencies,
            environment=environment or {},
            input_data=input_data,
            metadata={
                "inferred_tier": explicit_tier is None,
                "code_length": len(code),
                "dependency_count": len(dependencies),
            },
        )

        self._logger.info(
            "job_created",
            job_id=str(job.id),
            user_id=user_id,
            tier=tier.value,
            timeout=timeout,
        )

        return job

    def validate_code(self, code: str, language: str = "python") -> tuple[bool, str | None]:
        """
        Basic validation of code before execution.

        Returns:
            (valid, error_message) tuple
        """
        if not code.strip():
            return False, "Empty code"

        if len(code) > 100000:
            return False, "Code too long (max 100KB)"

        # Check for obviously dangerous patterns
        dangerous_patterns = [
            r"\bos\.system\b",
            r"\bsubprocess\.(run|call|Popen)\b",
            r"\b__import__\b",
            r"\beval\s*\(",
            r"\bexec\s*\(",
            r"\bopen\s*\([^)]*['\"]w['\"]",
            r"\brm\s+-rf\b",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                self._logger.warning("dangerous_pattern_detected", pattern=pattern)
                # Note: We warn but don't block - the sandbox handles isolation
                break

        return True, None
