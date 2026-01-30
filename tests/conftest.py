"""
Pytest Configuration and Fixtures for DOVA Tests.
"""

import asyncio
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest

from dova.agents.base import AgentResult, AgentTask
from dova.config.providers import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMRouter,
    ModelConfig,
    ProviderConfig,
    TaskType,
)
from dova.config.settings import Settings
from dova.utils.cache import InMemoryCache
from dova.utils.metrics import MetricsCollector


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings(
        environment="development",
        log_level="DEBUG",
    )


@pytest.fixture
def mock_llm_response() -> LLMResponse:
    """Create mock LLM response."""
    return LLMResponse(
        content="Test response from LLM",
        provider="mock",
        model="mock-model",
        input_tokens=10,
        output_tokens=20,
        latency_ms=100.0,
    )


@pytest.fixture
def mock_llm_provider(mock_llm_response: LLMResponse) -> LLMProvider:
    """Create mock LLM provider."""
    provider = MagicMock(spec=LLMProvider)
    provider.name = "mock"
    provider.config = ProviderConfig(
        name="mock",
        enabled=True,
        priority=1,
        models={
            TaskType.REASONING: ModelConfig(model_id="mock-model"),
            TaskType.SUMMARIZATION: ModelConfig(model_id="mock-model"),
            TaskType.CLASSIFICATION: ModelConfig(model_id="mock-model"),
        },
    )

    async def mock_complete(request: LLMRequest) -> LLMResponse:
        return mock_llm_response

    provider.complete = AsyncMock(side_effect=mock_complete)
    provider.health_check = AsyncMock(return_value=True)

    return provider


@pytest.fixture
def mock_llm_router(mock_llm_provider: LLMProvider) -> LLMRouter:
    """Create mock LLM router."""
    router = LLMRouter(
        providers={"mock": mock_llm_provider},
    )
    return router


@pytest.fixture
def mock_mcp_client() -> AsyncMock:
    """Create mock MCP client."""
    client = AsyncMock()

    async def mock_invoke(
        server: str, tool: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        if server == "arxiv":
            return [
                {
                    "title": "Test Paper",
                    "id": "arxiv:1234.5678",
                    "summary": "Test abstract",
                    "authors": ["Author One", "Author Two"],
                }
            ]
        elif server == "github":
            return {
                "items": [
                    {
                        "full_name": "test/repo",
                        "description": "Test repository",
                        "stargazers_count": 100,
                        "html_url": "https://github.com/test/repo",
                    }
                ]
            }
        elif server == "huggingface":
            return [
                {
                    "id": "test/model",
                    "downloads": 1000,
                    "pipeline_tag": "text-generation",
                }
            ]
        return {}

    client.invoke = AsyncMock(side_effect=mock_invoke)
    return client


@pytest.fixture
def metrics_collector() -> MetricsCollector:
    """Create test metrics collector."""
    return MetricsCollector(service_name="dova-test", enable_otel=False)


@pytest.fixture
def cache() -> InMemoryCache:
    """Create test cache."""
    return InMemoryCache(max_size=100)


@pytest.fixture
def sample_task() -> AgentTask:
    """Create sample agent task."""
    return AgentTask(
        id="test-task-123",
        type="research",
        params={"query": "test query"},
        user_id="test-user",
    )


@pytest.fixture
def sample_result() -> AgentResult:
    """Create sample agent result."""
    return AgentResult(
        success=True,
        data={"summary": "Test result"},
        agent_name="TestAgent",
        task_id="test-task-123",
    )


# Async fixtures for testing
@pytest.fixture
async def async_cache() -> AsyncGenerator[InMemoryCache, None]:
    """Create async test cache."""
    cache = InMemoryCache()
    yield cache
    await cache.clear()
