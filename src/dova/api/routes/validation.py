"""
Validation Endpoints for DOVA API.
"""

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from dova.api.middleware.auth import get_current_user, User

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.post("/validate")
async def validate_code(
    request: Request,
    code: str,
    language: str = "python",
    dimensions: list[str] | None = None,
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Validate code for quality and security.

    Args:
        code: Code to validate
        language: Programming language (default: python)
        dimensions: Validation dimensions to check (quality, security, reliability)

    Returns:
        Validation report with scores and issues
    """
    logger.info(
        "validate_code",
        user_id=current_user.id,
        language=language,
        code_length=len(code),
    )

    if not code.strip():
        raise HTTPException(status_code=400, detail="No code provided")

    if len(code) > 50000:
        raise HTTPException(status_code=400, detail="Code exceeds maximum length (50000 chars)")

    try:
        validation_agent = getattr(request.app.state, "validation_agent", None)

        if validation_agent is None:
            return {
                "status": "warning",
                "message": "Validation agent not initialized",
                "overall_score": 0,
                "passed": False,
                "issues": [],
            }

        from dova.agents.base import AgentTask
        from dova.agents.validation import ValidationDimension

        # Parse dimensions
        parsed_dimensions = []
        if dimensions:
            for dim in dimensions:
                try:
                    parsed_dimensions.append(ValidationDimension(dim))
                except ValueError:
                    pass

        if not parsed_dimensions:
            parsed_dimensions = list(ValidationDimension)

        task = AgentTask(
            type="analyze",
            params={
                "code": code,
                "language": language,
                "dimensions": parsed_dimensions,
            },
            user_id=current_user.id,
        )

        result = await validation_agent.execute(task)

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)

        return result.data

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("validate_code_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate/security")
async def security_scan(
    request: Request,
    code: str,
    language: str = "python",
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Run security-focused validation on code.

    Args:
        code: Code to scan
        language: Programming language

    Returns:
        Security scan results with vulnerabilities
    """
    logger.info(
        "security_scan",
        user_id=current_user.id,
        language=language,
        code_length=len(code),
    )

    if not code.strip():
        raise HTTPException(status_code=400, detail="No code provided")

    try:
        validation_agent = getattr(request.app.state, "validation_agent", None)

        if validation_agent is None:
            return {
                "status": "warning",
                "message": "Validation agent not initialized",
                "vulnerabilities": [],
                "risk_level": "unknown",
            }

        from dova.agents.base import AgentTask
        from dova.agents.validation import ValidationDimension

        task = AgentTask(
            type="analyze",
            params={
                "code": code,
                "language": language,
                "dimensions": [ValidationDimension.SECURITY],
            },
            user_id=current_user.id,
        )

        result = await validation_agent.execute(task)

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)

        # Extract security-specific results
        data = result.data or {}
        security_issues = [
            i for i in data.get("issues", [])
            if i.get("dimension") == "security"
        ]

        # Determine risk level
        if any(i.get("severity") == "critical" for i in security_issues):
            risk_level = "critical"
        elif any(i.get("severity") == "high" for i in security_issues):
            risk_level = "high"
        elif any(i.get("severity") == "medium" for i in security_issues):
            risk_level = "medium"
        elif security_issues:
            risk_level = "low"
        else:
            risk_level = "none"

        return {
            "status": "completed",
            "vulnerabilities": security_issues,
            "risk_level": risk_level,
            "security_score": data.get("dimension_scores", {}).get("security", 100),
            "recommendations": [
                r for r in data.get("recommendations", [])
                if "security" in r.lower() or "vulnerab" in r.lower()
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("security_scan_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate/full")
async def full_validation(
    request: Request,
    code: str,
    language: str = "python",
    execute: bool = False,
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Run full validation including all dimensions and optional execution.

    Args:
        code: Code to validate
        language: Programming language
        execute: Whether to execute code in sandbox (requires special permissions)

    Returns:
        Comprehensive validation report
    """
    logger.info(
        "full_validation",
        user_id=current_user.id,
        language=language,
        execute=execute,
        code_length=len(code),
    )

    if not code.strip():
        raise HTTPException(status_code=400, detail="No code provided")

    if execute:
        # Check if sandbox is enabled
        settings = getattr(request.app.state, "settings", None)
        if not settings or not settings.sandbox.enabled:
            raise HTTPException(
                status_code=403,
                detail="Sandbox execution not enabled",
            )

    try:
        validation_agent = getattr(request.app.state, "validation_agent", None)

        if validation_agent is None:
            return {
                "status": "warning",
                "message": "Validation agent not initialized",
            }

        from dova.agents.base import AgentTask

        task = AgentTask(
            type="full_validation",
            params={
                "code": code,
                "language": language,
            },
            user_id=current_user.id,
        )

        result = await validation_agent.execute(task)

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)

        return {
            "status": "completed",
            **result.data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("full_validation_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate/execute")
async def execute_code(
    request: Request,
    code: str,
    language: str = "python",
    dependencies: list[str] | None = None,
    tier: str | None = None,
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Execute code in a sandboxed environment.

    Args:
        code: Code to execute
        language: Programming language (python, node, go, rust)
        dependencies: List of packages to install
        tier: Resource tier (cpu_basic, cpu_standard, gpu_spot, gpu_premium)

    Returns:
        Execution result with output
    """
    logger.info(
        "execute_code",
        user_id=current_user.id,
        language=language,
        code_length=len(code),
        tier=tier,
    )

    if not code.strip():
        raise HTTPException(status_code=400, detail="No code provided")

    if len(code) > 100000:
        raise HTTPException(status_code=400, detail="Code exceeds maximum length (100KB)")

    # Check if sandbox is enabled
    settings = getattr(request.app.state, "settings", None)
    if not settings or not settings.sandbox.enabled:
        raise HTTPException(
            status_code=503,
            detail="Sandbox execution not enabled",
        )

    sandbox_executor = getattr(request.app.state, "sandbox_executor", None)
    sandbox_scheduler = getattr(request.app.state, "sandbox_scheduler", None)
    quota_manager = getattr(request.app.state, "quota_manager", None)

    if not sandbox_executor or not sandbox_scheduler:
        raise HTTPException(
            status_code=503,
            detail="Sandbox services not initialized",
        )

    try:
        from dova.services.sandbox.types import SandboxTier

        # Parse tier if provided
        explicit_tier = None
        if tier:
            try:
                explicit_tier = SandboxTier(tier)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid tier: {tier}. Valid: cpu_basic, cpu_standard, gpu_spot, gpu_premium",
                )

        # Create job
        job = sandbox_scheduler.create_job(
            user_id=current_user.id,
            code=code,
            language=language,
            dependencies=dependencies or [],
            explicit_tier=explicit_tier,
        )

        # Check quota
        if quota_manager:
            allowed, reason = await quota_manager.check_quota(
                current_user.id,
                job.tier,
                job.timeout_seconds,
            )
            if not allowed:
                raise HTTPException(status_code=429, detail=f"Quota exceeded: {reason}")

            await quota_manager.reserve_quota(current_user.id, job.tier, job.timeout_seconds)

        # Execute
        result = await sandbox_executor.execute(job)

        # Record actual usage
        if quota_manager:
            await quota_manager.record_usage(
                current_user.id,
                job.tier,
                result.execution_time_seconds,
                reserved_seconds=job.timeout_seconds,
            )

        return {
            "status": "completed" if result.success else "failed",
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "exit_code": result.exit_code,
            "execution_time_seconds": result.execution_time_seconds,
            "tier_used": result.tier_used.value,
            "job_id": str(job.id),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("execute_code_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validate/quota")
async def get_quota(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Get user's remaining sandbox execution quota.

    Returns:
        Remaining CPU and GPU quota in seconds
    """
    quota_manager = getattr(request.app.state, "quota_manager", None)
    if not quota_manager:
        return {
            "cpu_seconds": 0,
            "gpu_seconds": 0,
            "message": "Sandbox not enabled",
        }

    remaining = await quota_manager.get_remaining(current_user.id)
    return {
        "cpu_seconds": remaining["cpu_seconds"],
        "gpu_seconds": remaining["gpu_seconds"],
        "reset_in_hours": remaining["reset_in_hours"],
    }
