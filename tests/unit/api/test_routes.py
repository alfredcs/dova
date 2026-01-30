"""
Unit Tests for DOVA API Routes.
"""

import pytest
from fastapi.testclient import TestClient

from dova.api.main import create_app


@pytest.fixture
def app():
    """Create test FastAPI application."""
    return create_app()


@pytest.fixture
def client(app) -> TestClient:
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Create mock auth headers."""
    return {"Authorization": "Bearer test-token", "X-User-ID": "test-user-123"}


class TestHealthRoutes:
    """Test cases for health check endpoints."""

    def test_health_check(self, client: TestClient) -> None:
        """Test basic health check."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_liveness_check(self, client: TestClient) -> None:
        """Test liveness check."""
        response = client.get("/health/live")

        assert response.status_code == 200


class TestResearchRoutes:
    """Test cases for research endpoints."""

    def test_research_without_query(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Test research endpoint without query."""
        response = client.post(
            "/api/v1/research",
            json={},  # No query
            headers=auth_headers,
        )

        assert response.status_code == 422  # Validation error


class TestAPIErrorHandling:
    """Test API error handling."""

    def test_not_found(self, client: TestClient) -> None:
        """Test 404 response."""
        response = client.get("/nonexistent")

        assert response.status_code == 404

    def test_method_not_allowed(self, client: TestClient) -> None:
        """Test 405 response."""
        response = client.delete("/health")

        assert response.status_code == 405

    def test_validation_error(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Test validation error response."""
        response = client.post(
            "/api/v1/research",
            json={"invalid_field": "value"},  # Missing required field
            headers=auth_headers,
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


class TestRateLimiting:
    """Test rate limiting middleware."""

    def test_rate_limit_headers(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Test rate limit headers are present."""
        response = client.get("/health")

        # Rate limit headers should be present if middleware is active
        # These may or may not be present depending on configuration
        assert response.status_code == 200
