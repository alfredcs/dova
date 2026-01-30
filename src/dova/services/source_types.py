"""Source type definitions."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SourceType(Enum):
    BUILTIN = "builtin"      # arxiv, github, huggingface
    WEB_URL = "web_url"      # scrape web pages
    RSS_FEED = "rss_feed"    # RSS/Atom feeds
    API = "api"              # custom API endpoints


@dataclass
class SourceConfig:
    """Configuration for a source."""
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    auth_type: str | None = None  # "bearer", "api_key", "basic"
    auth_value: str | None = None
    refresh_interval_minutes: int = 60
    content_selector: str | None = None  # CSS selector for web scraping


@dataclass
class QualityMetrics:
    """Implicit quality signals for a source."""
    query_count: int = 0
    click_count: int = 0
    save_count: int = 0
    total_results: int = 0
    avg_position_clicked: float = 0.0  # lower = better
    last_used: datetime | None = None

    @property
    def quality_score(self) -> float:
        """Calculate quality score from implicit signals (0-1)."""
        if self.query_count == 0:
            return 0.5  # neutral for new sources

        click_rate = self.click_count / max(self.total_results, 1)
        save_rate = self.save_count / max(self.click_count, 1)
        position_score = 1.0 / (1 + self.avg_position_clicked / 10)

        # Weighted combination
        return min(1.0, (click_rate * 0.4) + (save_rate * 0.3) + (position_score * 0.3))


@dataclass
class Source:
    """A research source (built-in or custom)."""
    id: str
    user_id: str
    name: str
    source_type: SourceType
    enabled: bool = True
    config: SourceConfig | None = None
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "source_type": self.source_type.value,
            "enabled": self.enabled,
            "config": {
                "url": self.config.url,
                "headers": self.config.headers,
                "auth_type": self.config.auth_type,
                "refresh_interval_minutes": self.config.refresh_interval_minutes,
                "content_selector": self.config.content_selector,
            } if self.config else None,
            "quality": {
                "query_count": self.quality.query_count,
                "click_count": self.quality.click_count,
                "save_count": self.quality.save_count,
                "total_results": self.quality.total_results,
                "avg_position_clicked": self.quality.avg_position_clicked,
                "last_used": self.quality.last_used.isoformat() if self.quality.last_used else None,
            },
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Source":
        """Deserialize from storage."""
        config = None
        if data.get("config"):
            cfg = data["config"]
            config = SourceConfig(
                url=cfg["url"],
                headers=cfg.get("headers", {}),
                auth_type=cfg.get("auth_type"),
                auth_value=cfg.get("auth_value"),
                refresh_interval_minutes=cfg.get("refresh_interval_minutes", 60),
                content_selector=cfg.get("content_selector"),
            )

        q = data.get("quality", {})
        quality = QualityMetrics(
            query_count=q.get("query_count", 0),
            click_count=q.get("click_count", 0),
            save_count=q.get("save_count", 0),
            total_results=q.get("total_results", 0),
            avg_position_clicked=q.get("avg_position_clicked", 0.0),
            last_used=datetime.fromisoformat(q["last_used"]) if q.get("last_used") else None,
        )

        return cls(
            id=data["id"],
            user_id=data["user_id"],
            name=data["name"],
            source_type=SourceType(data["source_type"]),
            enabled=data.get("enabled", True),
            config=config,
            quality=quality,
            created_at=datetime.fromisoformat(data["created_at"]),
        )
