"""Webhook endpoints for external service integrations."""
import hashlib
import hmac
from typing import Any

import structlog
from fastapi import APIRouter, Header, HTTPException, Request

from dova.jobs.jobs import Job, JobType

router = APIRouter(prefix="/webhooks")
logger = structlog.get_logger(__name__)


@router.post("/github")
async def github_webhook(
    request: Request,
    x_github_event: str | None = Header(default=None, alias="X-GitHub-Event"),
    x_github_delivery: str | None = Header(default=None, alias="X-GitHub-Delivery"),
    x_hub_signature_256: str | None = Header(default=None, alias="X-Hub-Signature-256"),
) -> dict:
    """
    Receive GitHub webhook events.

    Handles:
    - push: New commits to watched repositories
    - release: New releases
    - star: Repository starred (trending signal)
    - issues: Issue opened/closed
    """
    body = await request.body()
    payload = await request.json()

    # Verify signature if webhook secret is configured
    settings = getattr(request.app.state, "settings", None)
    webhook_secret = getattr(settings, "github_webhook_secret", None) if settings else None

    if webhook_secret and x_hub_signature_256:
        if not _verify_github_signature(body, x_hub_signature_256, webhook_secret):
            logger.warning("github_webhook_invalid_signature", delivery_id=x_github_delivery)
            raise HTTPException(status_code=401, detail="Invalid signature")

    logger.info(
        "github_webhook_received",
        event=x_github_event,
        delivery_id=x_github_delivery,
        repo=payload.get("repository", {}).get("full_name"),
    )

    # Enqueue job for processing
    job_queue = getattr(request.app.state, "job_queue", None)
    if job_queue and x_github_event:
        job = Job(
            type=JobType.GITHUB_WEBHOOK,
            payload={
                "event": x_github_event,
                "delivery_id": x_github_delivery,
                "data": _extract_relevant_data(x_github_event, payload),
            },
        )
        await job_queue.enqueue(job)
        logger.debug("github_webhook_job_enqueued", job_id=str(job.id))

    return {"status": "received", "event": x_github_event}


def _verify_github_signature(body: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook signature."""
    if not signature.startswith("sha256="):
        return False

    expected_sig = signature[7:]
    computed_sig = hmac.new(
        secret.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected_sig, computed_sig)


def _extract_relevant_data(event: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Extract relevant data from webhook payload based on event type."""
    repo = payload.get("repository", {})
    sender = payload.get("sender", {})

    base_data = {
        "repo_name": repo.get("full_name"),
        "repo_url": repo.get("html_url"),
        "sender": sender.get("login"),
    }

    if event == "push":
        return {
            **base_data,
            "ref": payload.get("ref"),
            "commits": [
                {
                    "sha": c.get("id", "")[:7],
                    "message": c.get("message", "").split("\n")[0],
                    "author": c.get("author", {}).get("name"),
                }
                for c in payload.get("commits", [])[:5]
            ],
        }

    elif event == "release":
        release = payload.get("release", {})
        return {
            **base_data,
            "action": payload.get("action"),
            "tag_name": release.get("tag_name"),
            "name": release.get("name"),
            "prerelease": release.get("prerelease"),
            "body": release.get("body", "")[:500],
        }

    elif event == "star":
        return {
            **base_data,
            "action": payload.get("action"),
            "starred_at": payload.get("starred_at"),
            "stargazers_count": repo.get("stargazers_count"),
        }

    elif event == "issues":
        issue = payload.get("issue", {})
        return {
            **base_data,
            "action": payload.get("action"),
            "issue_number": issue.get("number"),
            "issue_title": issue.get("title"),
            "issue_url": issue.get("html_url"),
            "labels": [l.get("name") for l in issue.get("labels", [])],
        }

    elif event == "pull_request":
        pr = payload.get("pull_request", {})
        return {
            **base_data,
            "action": payload.get("action"),
            "pr_number": pr.get("number"),
            "pr_title": pr.get("title"),
            "pr_url": pr.get("html_url"),
            "merged": pr.get("merged", False),
        }

    else:
        return base_data
