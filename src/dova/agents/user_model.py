"""
User Model for DOVA ThinkingOrchestrator.

Rich user representation that captures:
- Expertise levels across topics
- Communication preferences
- Session context and goals
- Interaction patterns
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ExpertiseLevel(Enum):
    """User's expertise level in a topic."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"
    UNKNOWN = "unknown"


class ResponseDepth(Enum):
    """Preferred depth of responses."""

    BRIEF = "brief"  # Concise, to the point
    STANDARD = "standard"  # Balanced detail
    DETAILED = "detailed"  # In-depth explanations


@dataclass
class UserModel:
    """
    Rich user representation for personalized orchestration.

    Captures user characteristics beyond simple topic affinities
    to enable truly personalized responses.
    """

    user_id: str

    # Expertise (topic -> level)
    expertise_areas: dict[str, ExpertiseLevel] = field(default_factory=dict)

    # Communication preferences
    preferred_depth: ResponseDepth = ResponseDepth.STANDARD
    prefers_code_examples: bool = True
    prefers_citations: bool = True
    formality: str = "technical"  # "technical", "casual", "formal"

    # Session context
    current_goals: list[str] = field(default_factory=list)
    entities_of_interest: dict[str, Any] = field(default_factory=dict)

    # Inferred from interactions
    question_patterns: dict[str, int] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def get_expertise(self, topic: str) -> ExpertiseLevel:
        """Get expertise level for a topic, defaulting to UNKNOWN."""
        # Check exact match
        if topic in self.expertise_areas:
            return self.expertise_areas[topic]

        # Check partial matches
        topic_lower = topic.lower()
        for known_topic, level in self.expertise_areas.items():
            if known_topic.lower() in topic_lower or topic_lower in known_topic.lower():
                return level

        return ExpertiseLevel.UNKNOWN

    def update_expertise(self, topic: str, level: ExpertiseLevel) -> None:
        """Update expertise for a topic."""
        self.expertise_areas[topic] = level
        self.last_updated = datetime.utcnow()

    def add_goal(self, goal: str) -> None:
        """Add a session goal if not already present."""
        if goal not in self.current_goals:
            self.current_goals.append(goal)
            self.last_updated = datetime.utcnow()

    def record_question_type(self, question_type: str) -> None:
        """Record a question pattern for future inference."""
        self.question_patterns[question_type] = self.question_patterns.get(question_type, 0) + 1
        self.last_updated = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "user_id": self.user_id,
            "expertise_areas": {k: v.value for k, v in self.expertise_areas.items()},
            "preferred_depth": self.preferred_depth.value,
            "prefers_code_examples": self.prefers_code_examples,
            "prefers_citations": self.prefers_citations,
            "formality": self.formality,
            "current_goals": self.current_goals,
            "entities_of_interest": self.entities_of_interest,
            "question_patterns": self.question_patterns,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserModel":
        """Deserialize from dictionary."""
        expertise = {}
        for topic, level in data.get("expertise_areas", {}).items():
            try:
                expertise[topic] = ExpertiseLevel(level)
            except ValueError:
                expertise[topic] = ExpertiseLevel.UNKNOWN

        try:
            depth = ResponseDepth(data.get("preferred_depth", "standard"))
        except ValueError:
            depth = ResponseDepth.STANDARD

        return cls(
            user_id=data["user_id"],
            expertise_areas=expertise,
            preferred_depth=depth,
            prefers_code_examples=data.get("prefers_code_examples", True),
            prefers_citations=data.get("prefers_citations", True),
            formality=data.get("formality", "technical"),
            current_goals=data.get("current_goals", []),
            entities_of_interest=data.get("entities_of_interest", {}),
            question_patterns=data.get("question_patterns", {}),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else datetime.utcnow()
            ),
            last_updated=(
                datetime.fromisoformat(data["last_updated"])
                if data.get("last_updated")
                else datetime.utcnow()
            ),
        )

    def __repr__(self) -> str:
        expertise_summary = ", ".join(
            f"{t}:{l.value}" for t, l in list(self.expertise_areas.items())[:3]
        )
        return f"UserModel(user_id={self.user_id!r}, expertise=[{expertise_summary}], depth={self.preferred_depth.value})"
