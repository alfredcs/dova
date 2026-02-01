"""
Session freshness and state management service.

Manages user sessions with automatic staleness detection and recovery.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

from dova.utils.cache import Cache

logger = structlog.get_logger(__name__)


class SessionState(Enum):
    """Session lifecycle states."""

    ACTIVE = "active"
    STALE = "stale"
    EXPIRED = "expired"


class SessionAction(Enum):
    """Actions to take based on session state."""

    CONTINUE = "continue"  # Session is fresh, continue normally
    REFRESH = "refresh"  # Session is stale, refresh context
    FORK = "fork"  # Create new session from current state
    REPAIR = "repair"  # Attempt to repair session state


@dataclass
class Session:
    """User session with context and state tracking."""

    id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    context: dict[str, Any] = field(default_factory=dict)
    state: SessionState = SessionState.ACTIVE

    def to_dict(self) -> dict[str, Any]:
        """Serialize session to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "context": self.context,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Deserialize session from dictionary."""
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            context=data.get("context", {}),
            state=SessionState(data.get("state", "active")),
        )


class SessionManager:
    """
    Manages user sessions with automatic freshness evaluation.

    Tracks session activity, detects staleness, and provides
    recovery strategies for expired or corrupted sessions.
    """

    def __init__(
        self,
        cache: Cache,
        stale_after_seconds: int = 1800,  # 30 minutes
        expire_after_seconds: int = 86400,  # 24 hours
    ):
        self.cache = cache
        self.stale_after_seconds = stale_after_seconds
        self.expire_after_seconds = expire_after_seconds
        self._logger = logger.bind(service="session")

    async def create_session(
        self,
        user_id: str,
        context: dict[str, Any] | None = None,
    ) -> Session:
        """
        Create a new session for a user.

        Args:
            user_id: User identifier
            context: Initial context data

        Returns:
            New Session instance
        """
        now = datetime.utcnow()
        session = Session(
            id=str(uuid4()),
            user_id=user_id,
            created_at=now,
            last_activity=now,
            context=context or {},
            state=SessionState.ACTIVE,
        )

        await self._store_session(session)

        self._logger.info(
            "session_created",
            session_id=session.id,
            user_id=user_id,
        )

        return session

    async def get_session(self, session_id: str) -> Session | None:
        """
        Retrieve a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session if found, None otherwise
        """
        cache_key = self._session_key(session_id)
        data = await self.cache.get(cache_key)

        if data is None:
            return None

        return Session.from_dict(data)

    async def get_user_sessions(self, user_id: str) -> list[Session]:
        """
        Get all sessions for a user.

        Args:
            user_id: User identifier

        Returns:
            List of sessions
        """
        index_key = self._user_sessions_key(user_id)
        session_ids = await self.cache.get(index_key) or []

        sessions = []
        for sid in session_ids:
            session = await self.get_session(sid)
            if session:
                sessions.append(session)

        return sessions

    async def update_activity(self, session_id: str) -> Session | None:
        """
        Update session last activity timestamp.

        Args:
            session_id: Session identifier

        Returns:
            Updated session or None if not found
        """
        session = await self.get_session(session_id)
        if session is None:
            return None

        session.last_activity = datetime.utcnow()
        session.state = SessionState.ACTIVE

        await self._store_session(session)
        return session

    async def update_context(
        self,
        session_id: str,
        context_update: dict[str, Any],
        merge: bool = True,
    ) -> Session | None:
        """
        Update session context.

        Args:
            session_id: Session identifier
            context_update: Context data to update
            merge: If True, merge with existing context; otherwise replace

        Returns:
            Updated session or None if not found
        """
        session = await self.get_session(session_id)
        if session is None:
            return None

        if merge:
            session.context.update(context_update)
        else:
            session.context = context_update

        session.last_activity = datetime.utcnow()
        await self._store_session(session)

        return session

    def evaluate_freshness(
        self,
        session: Session,
    ) -> tuple[SessionState, SessionAction]:
        """
        Evaluate session freshness and recommend action.

        Args:
            session: Session to evaluate

        Returns:
            Tuple of (SessionState, SessionAction)
        """
        now = datetime.utcnow()
        age = (now - session.last_activity).total_seconds()

        if age > self.expire_after_seconds:
            return SessionState.EXPIRED, SessionAction.FORK

        if age > self.stale_after_seconds:
            return SessionState.STALE, SessionAction.REFRESH

        return SessionState.ACTIVE, SessionAction.CONTINUE

    async def execute_action(
        self,
        session: Session,
        action: SessionAction,
    ) -> Session:
        """
        Execute a recovery action on a session.

        Args:
            session: Session to act on
            action: Action to execute

        Returns:
            Updated or new session
        """
        if action == SessionAction.CONTINUE:
            return session

        if action == SessionAction.REFRESH:
            session.last_activity = datetime.utcnow()
            session.state = SessionState.ACTIVE
            # Keep context but mark as refreshed
            session.context["_refreshed_at"] = datetime.utcnow().isoformat()
            await self._store_session(session)
            self._logger.info("session_refreshed", session_id=session.id)
            return session

        if action == SessionAction.FORK:
            # Create new session preserving some context
            preserved_context = {
                k: v for k, v in session.context.items()
                if not k.startswith("_")
            }
            preserved_context["_forked_from"] = session.id

            new_session = await self.create_session(
                user_id=session.user_id,
                context=preserved_context,
            )
            self._logger.info(
                "session_forked",
                old_session_id=session.id,
                new_session_id=new_session.id,
            )
            return new_session

        if action == SessionAction.REPAIR:
            # Attempt to repair by clearing potentially corrupted state
            session.context = {
                k: v for k, v in session.context.items()
                if not k.startswith("_error")
            }
            session.context["_repaired_at"] = datetime.utcnow().isoformat()
            session.state = SessionState.ACTIVE
            session.last_activity = datetime.utcnow()
            await self._store_session(session)
            self._logger.info("session_repaired", session_id=session.id)
            return session

        return session

    async def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions from all users.

        Returns:
            Count of sessions cleaned up
        """
        # This is a simplified implementation
        # In production, you'd scan Redis keys or maintain an index
        cleaned = 0
        self._logger.info("session_cleanup_started")

        # Note: Full implementation would iterate through all sessions
        # For now, this just logs the intent
        self._logger.info("session_cleanup_complete", cleaned=cleaned)
        return cleaned

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        session = await self.get_session(session_id)
        if session is None:
            return False

        cache_key = self._session_key(session_id)
        await self.cache.delete(cache_key)

        # Update user session index
        index_key = self._user_sessions_key(session.user_id)
        session_ids = await self.cache.get(index_key) or []
        if session_id in session_ids:
            session_ids.remove(session_id)
            await self.cache.set(index_key, session_ids, ttl=self.expire_after_seconds * 2)

        self._logger.info("session_deleted", session_id=session_id)
        return True

    def _session_key(self, session_id: str) -> str:
        """Generate cache key for a session."""
        return f"session:{session_id}"

    def _user_sessions_key(self, user_id: str) -> str:
        """Generate cache key for user session index."""
        return f"user_sessions:{user_id}"

    async def _store_session(self, session: Session) -> None:
        """Store session in cache."""
        cache_key = self._session_key(session.id)
        await self.cache.set(
            cache_key,
            session.to_dict(),
            ttl=self.expire_after_seconds * 1.5,  # Buffer beyond expiry
        )

        # Update user session index
        index_key = self._user_sessions_key(session.user_id)
        session_ids = await self.cache.get(index_key) or []
        if session.id not in session_ids:
            session_ids.append(session.id)
            await self.cache.set(index_key, session_ids, ttl=self.expire_after_seconds * 2)
