"""
Docker-based sandbox executor.
"""

import asyncio
import base64
import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from dova.services.sandbox.types import (
    ExecutionStatus,
    SandboxJob,
    SandboxTier,
    TierConfig,
)

logger = structlog.get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of sandbox execution."""

    success: bool
    output: str = ""
    error: str | None = None
    exit_code: int = 0
    execution_time_seconds: float = 0.0
    tier_used: SandboxTier = SandboxTier.CPU_BASIC
    metadata: dict[str, Any] = field(default_factory=dict)


# Base images for different languages
LANGUAGE_IMAGES = {
    "python": "python:3.11-slim",
    "python3": "python:3.11-slim",
    "node": "node:20-slim",
    "javascript": "node:20-slim",
    "go": "golang:1.22-alpine",
    "rust": "rust:1.75-slim",
}


class DockerExecutor:
    """
    Execute code in isolated Docker containers.

    Provides secure sandboxing with resource limits.
    """

    def __init__(
        self,
        docker_host: str = "unix:///var/run/docker.sock",
        network_enabled: bool = False,
        max_output_size: int = 100000,
    ):
        self.docker_host = docker_host
        self.network_enabled = network_enabled
        self.max_output_size = max_output_size
        self._logger = logger.bind(service="docker_executor")

    async def execute(self, job: SandboxJob) -> ExecutionResult:
        """
        Execute a sandbox job in a Docker container.

        Args:
            job: The job to execute

        Returns:
            ExecutionResult with output and status
        """
        start_time = datetime.utcnow()
        job.status = ExecutionStatus.RUNNING
        job.started_at = start_time

        config = TierConfig.get_config(job.tier)
        image = self._get_image(job.language)

        try:
            # Create container and execute
            result = await self._run_container(job, config, image)

            job.status = ExecutionStatus.COMPLETED
            job.output = result.output
            job.exit_code = result.exit_code

        except asyncio.TimeoutError:
            job.status = ExecutionStatus.TIMEOUT
            job.error = f"Execution timed out after {job.timeout_seconds}s"
            result = ExecutionResult(
                success=False,
                error=job.error,
                exit_code=-1,
                tier_used=job.tier,
            )

        except Exception as e:
            job.status = ExecutionStatus.FAILED
            job.error = str(e)
            self._logger.exception("execution_error", job_id=str(job.id), error=str(e))
            result = ExecutionResult(
                success=False,
                error=str(e),
                exit_code=-1,
                tier_used=job.tier,
            )

        end_time = datetime.utcnow()
        job.completed_at = end_time
        result.execution_time_seconds = (end_time - start_time).total_seconds()

        self._logger.info(
            "execution_complete",
            job_id=str(job.id),
            status=job.status.value,
            exit_code=job.exit_code,
            duration=result.execution_time_seconds,
        )

        return result

    async def _run_container(
        self,
        job: SandboxJob,
        config: TierConfig,
        image: str,
    ) -> ExecutionResult:
        """Run code in a Docker container using subprocess."""
        # Create temporary directory for code
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code to file
            code_file = Path(tmpdir) / self._get_filename(job.language)
            code_file.write_text(job.code)

            # Write input data if provided
            if job.input_data:
                input_file = Path(tmpdir) / "input.json"
                input_file.write_text(json.dumps(job.input_data))

            # Build docker command
            cmd = self._build_docker_command(job, config, image, tmpdir)

            # Execute with timeout
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=job.timeout_seconds,
                )

                output = stdout.decode("utf-8", errors="replace")
                error_output = stderr.decode("utf-8", errors="replace")

                # Truncate if needed
                if len(output) > self.max_output_size:
                    output = output[: self.max_output_size] + "\n... (truncated)"

                # Combine stderr into error if process failed
                if process.returncode != 0 and error_output:
                    error_output = error_output[: self.max_output_size]
                else:
                    error_output = None

                return ExecutionResult(
                    success=process.returncode == 0,
                    output=output,
                    error=error_output,
                    exit_code=process.returncode or 0,
                    tier_used=job.tier,
                )

            except asyncio.TimeoutError:
                # Try to kill the container
                await self._cleanup_container(job)
                raise

    def _build_docker_command(
        self,
        job: SandboxJob,
        config: TierConfig,
        image: str,
        tmpdir: str,
    ) -> list[str]:
        """Build the docker run command."""
        cmd = [
            "docker",
            "run",
            "--rm",
            # Resource limits
            f"--cpus={config.cpu_limit}",
            f"--memory={config.memory_mb}m",
            "--memory-swap=-1",  # No swap
            # Security
            "--security-opt=no-new-privileges",
            "--cap-drop=ALL",
            "--read-only",
            # Tmpfs for writable areas
            "--tmpfs=/tmp:size=100M",
            # Mount code directory
            f"-v={tmpdir}:/code:ro",
            "-w=/code",
        ]

        # Network isolation
        if not self.network_enabled:
            cmd.append("--network=none")

        # Container name for cleanup
        container_name = f"dova-sandbox-{job.id.hex[:12]}"
        cmd.extend(["--name", container_name])

        # Environment variables
        for key, value in job.environment.items():
            # Sanitize env vars
            if self._is_safe_env_var(key):
                cmd.extend(["-e", f"{key}={value}"])

        # Image
        cmd.append(image)

        # Command to run
        cmd.extend(self._get_run_command(job.language, job.dependencies))

        return cmd

    def _get_run_command(self, language: str, dependencies: list[str]) -> list[str]:
        """Get the command to run inside container."""
        if language in ("python", "python3"):
            if dependencies:
                # Install deps and run
                deps_str = " ".join(dependencies)
                return [
                    "sh",
                    "-c",
                    f"pip install --quiet {deps_str} && python main.py",
                ]
            return ["python", "main.py"]

        elif language in ("node", "javascript"):
            if dependencies:
                deps_str = " ".join(dependencies)
                return [
                    "sh",
                    "-c",
                    f"npm install --silent {deps_str} && node main.js",
                ]
            return ["node", "main.js"]

        elif language == "go":
            return ["go", "run", "main.go"]

        elif language == "rust":
            return ["sh", "-c", "rustc main.rs -o /tmp/main && /tmp/main"]

        return ["sh", "-c", "echo 'Unsupported language'"]

    def _get_filename(self, language: str) -> str:
        """Get appropriate filename for language."""
        extensions = {
            "python": "main.py",
            "python3": "main.py",
            "node": "main.js",
            "javascript": "main.js",
            "go": "main.go",
            "rust": "main.rs",
        }
        return extensions.get(language, "main.txt")

    def _get_image(self, language: str) -> str:
        """Get Docker image for language."""
        return LANGUAGE_IMAGES.get(language, "python:3.11-slim")

    def _is_safe_env_var(self, key: str) -> bool:
        """Check if environment variable name is safe."""
        # Block potentially dangerous env vars
        dangerous = {"PATH", "LD_PRELOAD", "LD_LIBRARY_PATH", "HOME", "USER"}
        return key.upper() not in dangerous and key.isalnum() or "_" in key

    async def _cleanup_container(self, job: SandboxJob) -> None:
        """Force remove a container if it exists."""
        container_name = f"dova-sandbox-{job.id.hex[:12]}"
        try:
            process = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "-f",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await process.wait()
        except Exception:
            pass

    async def check_docker_available(self) -> bool:
        """Check if Docker is available and running."""
        try:
            process = await asyncio.create_subprocess_exec(
                "docker",
                "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await process.wait()
            return process.returncode == 0
        except Exception:
            return False
