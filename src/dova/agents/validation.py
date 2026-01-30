"""
Validation Agent for DOVA.

Provides code validation capabilities:
- Code quality analysis
- Security scanning
- Execution in sandboxed environments
- Performance benchmarking
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from dova.agents.base import AgentResult, AgentTask, BaseAgent
from dova.config.providers import LLMRouter, TaskType
from dova.utils.metrics import MetricsCollector

logger = structlog.get_logger(__name__)


class ValidationDimension(Enum):
    """Dimensions of code validation."""

    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    QUALITY = "quality"
    RELIABILITY = "reliability"


@dataclass
class ValidationIssue:
    """A single validation issue."""

    dimension: ValidationDimension
    severity: str  # "critical", "high", "medium", "low", "info"
    message: str
    location: str | None = None
    suggestion: str | None = None


@dataclass
class ValidationReport:
    """Complete validation report."""

    overall_score: float  # 0-100
    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    dimension_scores: dict[str, float] = field(default_factory=dict)
    execution_output: str | None = None
    recommendations: list[str] = field(default_factory=list)


class ValidationAgent(BaseAgent):
    """
    Validation Agent for code quality and execution validation.

    Analyzes code for quality, security, and can execute
    in sandboxed environments for functional validation.
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        mcp_client: Any | None = None,
        metrics: MetricsCollector | None = None,
        sandbox_enabled: bool = False,
        sandbox_executor: Any | None = None,
        sandbox_scheduler: Any | None = None,
        quota_manager: Any | None = None,
    ):
        super().__init__(llm_router, mcp_client, metrics)
        self.sandbox_enabled = sandbox_enabled
        self.sandbox_executor = sandbox_executor
        self.sandbox_scheduler = sandbox_scheduler
        self.quota_manager = quota_manager

    @property
    def system_prompt(self) -> str:
        return """You are a Code Validation Agent responsible for analyzing code quality and security.

Your validation capabilities:
1. Static analysis - code quality, style, complexity
2. Security analysis - vulnerability detection, input validation
3. Reliability analysis - error handling, edge cases
4. Performance analysis - algorithmic efficiency, resource usage

When analyzing code:
- Identify potential bugs and logic errors
- Check for security vulnerabilities (OWASP Top 10)
- Evaluate code maintainability and readability
- Suggest improvements and best practices

Provide detailed, actionable feedback."""

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute validation task."""
        start_time = time.time()

        try:
            task_type = task.type
            code = task.params.get("code", "")
            language = task.params.get("language", "python")
            dimensions = task.params.get("dimensions", list(ValidationDimension))

            if not code:
                return self._wrap_result(task, False, error="No code provided")

            self._logger.info(
                "validation_starting",
                task_type=task_type,
                language=language,
                code_length=len(code),
            )

            if task_type == "analyze":
                report = await self._analyze_code(code, language, dimensions)
            elif task_type == "execute":
                if not self.sandbox_enabled:
                    return self._wrap_result(task, False, error="Sandbox execution not enabled")
                report = await self._execute_code(code, language, task.params)
            elif task_type == "full_validation":
                report = await self._full_validation(code, language, task.params)
            else:
                return self._wrap_result(task, False, error=f"Unknown task type: {task_type}")

            execution_time = (time.time() - start_time) * 1000
            return self._wrap_result(
                task,
                True,
                data={
                    "overall_score": report.overall_score,
                    "passed": report.passed,
                    "issues": [
                        {
                            "dimension": i.dimension.value,
                            "severity": i.severity,
                            "message": i.message,
                            "location": i.location,
                            "suggestion": i.suggestion,
                        }
                        for i in report.issues
                    ],
                    "dimension_scores": report.dimension_scores,
                    "recommendations": report.recommendations,
                    "execution_output": report.execution_output,
                },
                execution_time_ms=execution_time,
            )

        except Exception as e:
            self._logger.exception("validation_error", error=str(e))
            return self._wrap_result(
                task,
                False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _analyze_code(
        self,
        code: str,
        language: str,
        dimensions: list[ValidationDimension],
    ) -> ValidationReport:
        """Analyze code quality using LLM."""
        analysis_prompt = f"""Analyze this {language} code for quality and issues:

```{language}
{code}
```

Analyze the following dimensions: {[d.value for d in dimensions]}

For each issue found, provide:
- dimension: which validation dimension
- severity: critical/high/medium/low/info
- message: description of the issue
- location: line number or function name if applicable
- suggestion: how to fix it

Also provide:
- overall_score: 0-100 quality score
- dimension_scores: score for each dimension (0-100)
- recommendations: top 3 improvement suggestions

Respond in JSON format:
{{
    "overall_score": <number>,
    "passed": <true if score >= 70>,
    "issues": [
        {{
            "dimension": "<dimension>",
            "severity": "<severity>",
            "message": "<message>",
            "location": "<location>",
            "suggestion": "<suggestion>"
        }}
    ],
    "dimension_scores": {{
        "<dimension>": <score>
    }},
    "recommendations": ["<rec1>", "<rec2>", "<rec3>"]
}}"""

        response = await self.think(
            analysis_prompt,
            task_type=TaskType.REASONING,
            temperature=0.3,
        )

        return self._parse_analysis_response(response)

    async def _analyze_security(self, code: str, language: str) -> list[ValidationIssue]:
        """Analyze code for security vulnerabilities."""
        security_prompt = f"""Analyze this {language} code for security vulnerabilities:

```{language}
{code}
```

Check for:
1. Injection vulnerabilities (SQL, command, XSS)
2. Authentication/authorization issues
3. Sensitive data exposure
4. Insecure dependencies
5. Input validation issues
6. Cryptographic weaknesses
7. Error handling that reveals information

For each vulnerability found, provide:
- severity: critical/high/medium/low
- message: description
- location: where in the code
- suggestion: remediation

Respond as JSON array of issues."""

        response = await self.think(
            security_prompt,
            task_type=TaskType.REASONING,
            temperature=0.2,
        )

        issues = []
        import json

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            parsed = json.loads(response.strip())
            if isinstance(parsed, list):
                for item in parsed:
                    issues.append(
                        ValidationIssue(
                            dimension=ValidationDimension.SECURITY,
                            severity=item.get("severity", "medium"),
                            message=item.get("message", ""),
                            location=item.get("location"),
                            suggestion=item.get("suggestion"),
                        )
                    )
        except json.JSONDecodeError:
            self._logger.warning("security_analysis_parse_error")

        return issues

    async def _execute_code(
        self,
        code: str,
        language: str,
        params: dict[str, Any],
    ) -> ValidationReport:
        """Execute code in sandbox and validate results."""
        self._logger.info("sandbox_execution", language=language)

        # Check if sandbox services are available
        if not self.sandbox_executor or not self.sandbox_scheduler:
            return ValidationReport(
                overall_score=0,
                passed=False,
                issues=[
                    ValidationIssue(
                        dimension=ValidationDimension.FUNCTIONAL,
                        severity="info",
                        message="Sandbox execution not configured",
                    )
                ],
                execution_output="Sandbox execution requires additional configuration",
                recommendations=["Enable sandbox in settings (SANDBOX_ENABLED=true)"],
            )

        user_id = params.get("user_id", "anonymous")
        dependencies = params.get("dependencies", [])
        input_data = params.get("input_data")
        explicit_tier = params.get("tier")

        # Parse explicit tier if provided
        from dova.services.sandbox.types import SandboxTier

        tier = None
        if explicit_tier:
            try:
                tier = SandboxTier(explicit_tier)
            except ValueError:
                pass

        # Create sandbox job
        job = self.sandbox_scheduler.create_job(
            user_id=user_id,
            code=code,
            language=language,
            dependencies=dependencies,
            input_data=input_data,
            explicit_tier=tier,
        )

        # Check quota if quota manager is available
        if self.quota_manager:
            allowed, reason = await self.quota_manager.check_quota(
                user_id,
                job.tier,
                job.timeout_seconds,
            )
            if not allowed:
                return ValidationReport(
                    overall_score=0,
                    passed=False,
                    issues=[
                        ValidationIssue(
                            dimension=ValidationDimension.FUNCTIONAL,
                            severity="high",
                            message=f"Quota exceeded: {reason}",
                        )
                    ],
                    execution_output=None,
                    recommendations=["Wait for quota reset or upgrade your plan"],
                )

            # Reserve quota
            await self.quota_manager.reserve_quota(user_id, job.tier, job.timeout_seconds)

        # Execute in sandbox
        try:
            result = await self.sandbox_executor.execute(job)

            # Record actual usage
            if self.quota_manager:
                await self.quota_manager.record_usage(
                    user_id,
                    job.tier,
                    result.execution_time_seconds,
                    reserved_seconds=job.timeout_seconds,
                )

            # Build report from result
            if result.success:
                return ValidationReport(
                    overall_score=100,
                    passed=True,
                    issues=[],
                    execution_output=result.output,
                    recommendations=[],
                )
            else:
                return ValidationReport(
                    overall_score=30,
                    passed=False,
                    issues=[
                        ValidationIssue(
                            dimension=ValidationDimension.FUNCTIONAL,
                            severity="high",
                            message=f"Execution failed with exit code {result.exit_code}",
                            suggestion="Check the error output for details",
                        )
                    ],
                    execution_output=result.output or result.error,
                    recommendations=["Review error output and fix the code"],
                )

        except Exception as e:
            self._logger.exception("sandbox_execution_error", error=str(e))
            return ValidationReport(
                overall_score=0,
                passed=False,
                issues=[
                    ValidationIssue(
                        dimension=ValidationDimension.FUNCTIONAL,
                        severity="critical",
                        message=f"Execution error: {str(e)}",
                    )
                ],
                execution_output=None,
                recommendations=["Contact support if this persists"],
            )

    async def _full_validation(
        self,
        code: str,
        language: str,
        params: dict[str, Any],
    ) -> ValidationReport:
        """Run full validation including all dimensions."""
        import asyncio

        # Run analyses in parallel
        quality_task = self._analyze_code(
            code,
            language,
            [ValidationDimension.QUALITY, ValidationDimension.RELIABILITY],
        )
        security_task = self._analyze_security(code, language)

        quality_report, security_issues = await asyncio.gather(
            quality_task, security_task
        )

        # Merge security issues into report
        quality_report.issues.extend(security_issues)

        # Recalculate overall score with security
        if security_issues:
            critical_count = sum(1 for i in security_issues if i.severity == "critical")
            high_count = sum(1 for i in security_issues if i.severity == "high")
            security_penalty = (critical_count * 20) + (high_count * 10)
            quality_report.overall_score = max(0, quality_report.overall_score - security_penalty)
            quality_report.dimension_scores["security"] = max(0, 100 - security_penalty)
        else:
            quality_report.dimension_scores["security"] = 100

        quality_report.passed = quality_report.overall_score >= 70

        return quality_report

    def _parse_analysis_response(self, response: str) -> ValidationReport:
        """Parse LLM analysis response into ValidationReport."""
        import json

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            data = json.loads(response.strip())

            issues = []
            for item in data.get("issues", []):
                try:
                    dimension = ValidationDimension(item.get("dimension", "quality"))
                except ValueError:
                    dimension = ValidationDimension.QUALITY

                issues.append(
                    ValidationIssue(
                        dimension=dimension,
                        severity=item.get("severity", "medium"),
                        message=item.get("message", ""),
                        location=item.get("location"),
                        suggestion=item.get("suggestion"),
                    )
                )

            return ValidationReport(
                overall_score=data.get("overall_score", 50),
                passed=data.get("passed", False),
                issues=issues,
                dimension_scores=data.get("dimension_scores", {}),
                recommendations=data.get("recommendations", []),
            )

        except json.JSONDecodeError:
            self._logger.warning("analysis_parse_error", response=response[:200])
            return ValidationReport(
                overall_score=50,
                passed=False,
                issues=[
                    ValidationIssue(
                        dimension=ValidationDimension.QUALITY,
                        severity="info",
                        message="Could not parse detailed analysis",
                    )
                ],
                recommendations=["Re-run validation for detailed results"],
            )
