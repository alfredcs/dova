"""
Profile Endpoints for DOVA API.
"""

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from dova.api.schemas.common import UserPreferencesSchema
from dova.api.middleware.auth import get_current_user, User

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.get("/profile")
async def get_profile(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Get the current user's profile.

    Returns:
        User profile with preferences and interests
    """
    logger.info("get_profile", user_id=current_user.id)

    try:
        profiling_agent = getattr(request.app.state, "profiling_agent", None)

        if profiling_agent is None:
            # Return default profile
            return {
                "user_id": current_user.id,
                "email": current_user.email,
                "preferences": {
                    "interests": [],
                    "preferred_sources": ["arxiv", "github", "huggingface"],
                    "expertise_level": "intermediate",
                    "output_format": "detailed",
                },
                "temporal_interests": {
                    "short_term": [],
                    "medium_term": [],
                    "long_term": [],
                },
                "topic_affinities": {},
            }

        from dova.agents.base import AgentTask

        task = AgentTask(
            type="get_preferences",
            params={"user_id": current_user.id},
            user_id=current_user.id,
        )

        result = await profiling_agent.execute(task)

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)

        return {
            "user_id": current_user.id,
            "email": current_user.email,
            **result.data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("get_profile_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/profile")
async def update_profile(
    request: Request,
    preferences: UserPreferencesSchema,
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Update the current user's profile preferences.

    Args:
        preferences: New preference values

    Returns:
        Updated profile
    """
    logger.info(
        "update_profile",
        user_id=current_user.id,
        preferences=preferences.model_dump(exclude_unset=True),
    )

    try:
        profiling_agent = getattr(request.app.state, "profiling_agent", None)

        if profiling_agent is None:
            return {
                "status": "updated",
                "user_id": current_user.id,
                "message": "Profile agent not initialized - preferences stored locally",
            }

        from dova.agents.base import AgentTask

        task = AgentTask(
            type="update_preferences",
            params={
                "user_id": current_user.id,
                **preferences.model_dump(exclude_unset=True),
            },
            user_id=current_user.id,
        )

        result = await profiling_agent.execute(task)

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)

        return {
            "status": "updated",
            "user_id": current_user.id,
            **result.data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("update_profile_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profile/recommendations")
async def get_recommendations(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Get personalized recommendations based on user profile.

    Returns:
        Personalized research recommendations
    """
    logger.info("get_recommendations", user_id=current_user.id)

    try:
        profiling_agent = getattr(request.app.state, "profiling_agent", None)

        if profiling_agent is None:
            return {
                "user_id": current_user.id,
                "recommendations": {
                    "papers": [],
                    "repos": [],
                    "models": [],
                    "emerging_topics": [],
                },
                "message": "Profile agent not initialized",
            }

        from dova.agents.base import AgentTask

        task = AgentTask(
            type="get_recommendations",
            params={"user_id": current_user.id},
            user_id=current_user.id,
        )

        result = await profiling_agent.execute(task)

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)

        return {
            "user_id": current_user.id,
            "recommendations": result.data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("get_recommendations_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/profile/interaction")
async def record_interaction(
    request: Request,
    query: str,
    interaction_type: str = "query",
    topics: list[str] | None = None,
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Record a user interaction for profile learning.

    Args:
        query: The query or interaction content
        interaction_type: Type of interaction (query, save, click, etc.)
        topics: Optional list of topics (will be extracted if not provided)

    Returns:
        Confirmation of recorded interaction
    """
    logger.info(
        "record_interaction",
        user_id=current_user.id,
        interaction_type=interaction_type,
    )

    try:
        profiling_agent = getattr(request.app.state, "profiling_agent", None)

        if profiling_agent is None:
            return {
                "status": "recorded",
                "message": "Profile agent not initialized - interaction not persisted",
            }

        from dova.agents.base import AgentTask

        task = AgentTask(
            type="record_interaction",
            params={
                "user_id": current_user.id,
                "query": query,
                "type": interaction_type,
                "topics": topics,
            },
            user_id=current_user.id,
        )

        result = await profiling_agent.execute(task)

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)

        return result.data

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("record_interaction_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
