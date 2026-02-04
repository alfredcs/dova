"""
Conversation Context for DOVA ThinkingOrchestrator.

Session memory that tracks:
- Conversation turns with full metadata
- Entities discussed (papers, repos, models)
- Inferred goals and answered questions
- Topic continuity
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Metadata about this turn
    entities_mentioned: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)

    # For assistant turns
    action_taken: str | None = None
    action_rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "entities_mentioned": self.entities_mentioned,
            "tools_used": self.tools_used,
            "action_taken": self.action_taken,
            "action_rationale": self.action_rationale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationTurn":
        """Deserialize from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if data.get("timestamp")
                else datetime.utcnow()
            ),
            entities_mentioned=data.get("entities_mentioned", []),
            tools_used=data.get("tools_used", []),
            action_taken=data.get("action_taken"),
            action_rationale=data.get("action_rationale"),
        )


@dataclass
class ConversationContext:
    """
    Session context for conversation continuity.

    Tracks the full state of a conversation session including
    discussed entities, goals, and topic flow.
    """

    session_id: str
    user_id: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Conversation history
    turns: list[ConversationTurn] = field(default_factory=list)

    # Current topic being discussed
    current_topic: str = ""
    topic_history: list[str] = field(default_factory=list)

    # Entities discussed (full context, not just IDs)
    papers_discussed: list[dict[str, Any]] = field(default_factory=list)
    repos_discussed: list[dict[str, Any]] = field(default_factory=list)
    models_discussed: list[dict[str, Any]] = field(default_factory=list)

    # Goal tracking
    inferred_goals: list[str] = field(default_factory=list)
    questions_answered: dict[str, str] = field(default_factory=dict)

    # Pending state
    pending_questions: list[str] = field(default_factory=list)
    last_assistant_question: str = ""

    def add_turn(
        self,
        role: str,
        content: str,
        entities: list[str] | None = None,
        tools: list[str] | None = None,
        action: str | None = None,
        rationale: str | None = None,
    ) -> ConversationTurn:
        """Add a new turn to the conversation."""
        turn = ConversationTurn(
            role=role,
            content=content,
            entities_mentioned=entities or [],
            tools_used=tools or [],
            action_taken=action,
            action_rationale=rationale,
        )
        self.turns.append(turn)
        return turn

    def get_recent_turns(self, n: int = 6) -> list[ConversationTurn]:
        """Get the most recent n turns."""
        return self.turns[-n:] if len(self.turns) >= n else self.turns

    def get_conversation_summary(self, max_chars: int = 1000) -> str:
        """Get a summary of recent conversation for context."""
        lines = []
        for turn in self.get_recent_turns(4):
            prefix = "USER" if turn.role == "user" else "ASSISTANT"
            content = turn.content[:200] + "..." if len(turn.content) > 200 else turn.content
            lines.append(f"{prefix}: {content}")
        summary = "\n".join(lines)
        return summary[:max_chars] if len(summary) > max_chars else summary

    def update_topic(self, topic: str) -> None:
        """Update the current topic, tracking history."""
        if self.current_topic and self.current_topic != topic:
            if self.current_topic not in self.topic_history:
                self.topic_history.append(self.current_topic)
        self.current_topic = topic

    def add_paper(self, paper: dict[str, Any]) -> None:
        """Add a paper to the discussed entities."""
        # Check if already discussed (by title or ID)
        paper_id = paper.get("arxiv_id") or paper.get("id") or paper.get("title", "")
        for existing in self.papers_discussed:
            existing_id = existing.get("arxiv_id") or existing.get("id") or existing.get("title", "")
            if existing_id == paper_id:
                # Update existing entry
                existing.update(paper)
                return
        self.papers_discussed.append(paper)

    def add_repo(self, repo: dict[str, Any]) -> None:
        """Add a repository to the discussed entities."""
        repo_name = repo.get("name") or repo.get("full_name", "")
        for existing in self.repos_discussed:
            if existing.get("name") == repo_name or existing.get("full_name") == repo_name:
                existing.update(repo)
                return
        self.repos_discussed.append(repo)

    def add_model(self, model: dict[str, Any]) -> None:
        """Add a model to the discussed entities."""
        model_id = model.get("id") or model.get("modelId", "")
        for existing in self.models_discussed:
            if existing.get("id") == model_id:
                existing.update(model)
                return
        self.models_discussed.append(model)

    def get_entity_by_reference(self, reference: str) -> dict[str, Any] | None:
        """
        Find an entity by reference (e.g., "paper 1", "the first paper", "that repo").

        Returns the entity dict or None if not found.
        """
        ref_lower = reference.lower()

        # Check for numbered references like "paper 1", "first paper"
        ordinals = {
            "first": 0, "1": 0, "one": 0,
            "second": 1, "2": 1, "two": 1,
            "third": 2, "3": 2, "three": 2,
            "fourth": 3, "4": 3, "four": 3,
            "fifth": 4, "5": 4, "five": 4,
        }

        if "paper" in ref_lower:
            for ordinal, idx in ordinals.items():
                if ordinal in ref_lower:
                    if idx < len(self.papers_discussed):
                        return self.papers_discussed[idx]
            # Check for title match
            for paper in self.papers_discussed:
                title = paper.get("title", "").lower()
                if any(word in title for word in ref_lower.split() if len(word) > 3):
                    return paper

        elif "repo" in ref_lower or "repository" in ref_lower:
            for ordinal, idx in ordinals.items():
                if ordinal in ref_lower:
                    if idx < len(self.repos_discussed):
                        return self.repos_discussed[idx]
            # Check for name match
            for repo in self.repos_discussed:
                name = repo.get("name", "").lower()
                if any(word in name for word in ref_lower.split() if len(word) > 3):
                    return repo

        elif "model" in ref_lower:
            for ordinal, idx in ordinals.items():
                if ordinal in ref_lower:
                    if idx < len(self.models_discussed):
                        return self.models_discussed[idx]
            # Check for ID match
            for model in self.models_discussed:
                model_id = model.get("id", "").lower()
                if any(word in model_id for word in ref_lower.split() if len(word) > 3):
                    return model

        return None

    def record_answered_question(self, question: str, answer_summary: str) -> None:
        """Record that a question was answered."""
        self.questions_answered[question[:100]] = answer_summary[:200]

    def add_inferred_goal(self, goal: str) -> None:
        """Add an inferred goal if not already present."""
        if goal not in self.inferred_goals:
            self.inferred_goals.append(goal)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "turns": [t.to_dict() for t in self.turns],
            "current_topic": self.current_topic,
            "topic_history": self.topic_history,
            "papers_discussed": self.papers_discussed,
            "repos_discussed": self.repos_discussed,
            "models_discussed": self.models_discussed,
            "inferred_goals": self.inferred_goals,
            "questions_answered": self.questions_answered,
            "pending_questions": self.pending_questions,
            "last_assistant_question": self.last_assistant_question,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationContext":
        """Deserialize from dictionary."""
        ctx = cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else datetime.utcnow()
            ),
            current_topic=data.get("current_topic", ""),
            topic_history=data.get("topic_history", []),
            papers_discussed=data.get("papers_discussed", []),
            repos_discussed=data.get("repos_discussed", []),
            models_discussed=data.get("models_discussed", []),
            inferred_goals=data.get("inferred_goals", []),
            questions_answered=data.get("questions_answered", {}),
            pending_questions=data.get("pending_questions", []),
            last_assistant_question=data.get("last_assistant_question", ""),
        )
        ctx.turns = [ConversationTurn.from_dict(t) for t in data.get("turns", [])]
        return ctx

    def __repr__(self) -> str:
        return f"ConversationContext(session={self.session_id!r}, turns={len(self.turns)}, topic={self.current_topic!r})"
