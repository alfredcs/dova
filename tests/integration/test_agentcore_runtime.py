"""Integration tests for AgentCore runtime."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAgentCoreRuntime:
    """Integration tests for AgentCore runtime module."""

    @pytest.mark.integration
    def test_runtime_module_imports(self):
        """Test that runtime module can be imported."""
        # This should not raise ImportError even without bedrock-agentcore
        from dova.runtime import agentcore_app

        assert hasattr(agentcore_app, "agent_stream")
        assert hasattr(agentcore_app, "app")
        assert hasattr(agentcore_app, "format_response")

    @pytest.mark.integration
    def test_lazy_app_attribute_access(self):
        """Test that LazyApp properly wraps attribute access."""
        from dova.runtime.agentcore_app import LazyApp

        lazy = LazyApp()

        # Should not raise on attribute access (raises on actual use)
        assert hasattr(lazy, "run")
        assert hasattr(lazy, "entrypoint")

    @pytest.mark.integration
    def test_format_response_with_summary(self):
        """Test formatting response with summary data."""
        from dova.runtime.agentcore_app import format_response

        data = {
            "summary": "This is a test summary of the research findings.",
        }

        result = format_response(data)

        assert "## Summary" in result
        assert "This is a test summary" in result

    @pytest.mark.integration
    def test_format_response_with_insights(self):
        """Test formatting response with insights."""
        from dova.runtime.agentcore_app import format_response

        data = {
            "insights": [
                {"title": "First insight"},
                {"title": "Second insight"},
            ],
        }

        result = format_response(data)

        assert "## Key Findings" in result
        assert "First insight" in result
        assert "Second insight" in result

    @pytest.mark.integration
    def test_format_response_with_recommendations(self):
        """Test formatting response with recommendations."""
        from dova.runtime.agentcore_app import format_response

        data = {
            "recommendations": [
                {"action": "Do this first"},
                {"recommendation": "Then do this"},
            ],
        }

        result = format_response(data)

        assert "## Recommendations" in result
        assert "Do this first" in result
        assert "Then do this" in result

    @pytest.mark.integration
    def test_format_response_empty_data(self):
        """Test formatting response with empty data."""
        from dova.runtime.agentcore_app import format_response

        result = format_response(None)
        assert result == "No results found."

        # Empty dict also returns "No results found."
        result = format_response({})
        assert result == "No results found."

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_stream_basic(self):
        """Test basic agent_stream execution."""
        from dova.runtime.agentcore_app import agent_stream

        # Mock the orchestrator and its execute method
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.data = {"summary": "Test result"}

        mock_orchestrator = AsyncMock()
        mock_orchestrator.execute.return_value = mock_result

        with patch(
            "dova.runtime.agentcore_app.create_agent_with_memory",
            return_value=mock_orchestrator,
        ):
            payload = {
                "prompt": "Test query",
                "userId": "test-user",
                "runtimeSessionId": "test-session",
            }

            chunks = []
            async for chunk in agent_stream(payload):
                chunks.append(chunk)

            assert len(chunks) > 0
            assert "Test result" in chunks[0]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_stream_error_handling(self):
        """Test agent_stream error handling."""
        from dova.runtime.agentcore_app import agent_stream

        with patch(
            "dova.runtime.agentcore_app.create_agent_with_memory",
            side_effect=Exception("Test error"),
        ):
            payload = {
                "prompt": "Test query",
                "userId": "test-user",
                "runtimeSessionId": "test-session",
            }

            chunks = []
            async for chunk in agent_stream(payload):
                chunks.append(chunk)

            assert len(chunks) > 0
            assert "error" in chunks[0].lower()


class TestAgentCoreSettings:
    """Tests for AgentCore settings integration."""

    @pytest.mark.integration
    def test_agentcore_settings_exist(self):
        """Test that AgentCoreSettings can be loaded."""
        from dova.config.settings import get_settings

        settings = get_settings()

        assert hasattr(settings, "agentcore")
        assert hasattr(settings.agentcore, "stack_name")
        assert hasattr(settings.agentcore, "memory_id")
        assert hasattr(settings.agentcore, "gateway_url")
        assert hasattr(settings.agentcore, "runtime_mode")

    @pytest.mark.integration
    def test_memory_settings_exist(self):
        """Test that MemorySettings can be loaded."""
        from dova.config.settings import get_settings

        settings = get_settings()

        assert hasattr(settings, "memory")
        assert hasattr(settings.memory, "summary_enabled")
        assert hasattr(settings.memory, "user_preference_enabled")
        assert hasattr(settings.memory, "semantic_enabled")
        assert hasattr(settings.memory, "summary_top_k")
        assert hasattr(settings.memory, "semantic_relevance")

    @pytest.mark.integration
    def test_agentcore_settings_from_env(self):
        """Test AgentCoreSettings loads from environment."""
        from dova.config.settings import AgentCoreSettings

        with patch.dict(
            os.environ,
            {
                "AGENTCORE_STACK_NAME": "test-stack",
                "AGENTCORE_MEMORY_ID": "memory-123",
                "AGENTCORE_GATEWAY_URL": "https://gateway.test.com",
                "AGENTCORE_RUNTIME_MODE": "agentcore",
            },
        ):
            settings = AgentCoreSettings()

            assert settings.stack_name == "test-stack"
            assert settings.memory_id == "memory-123"
            assert settings.gateway_url == "https://gateway.test.com"
            assert settings.runtime_mode == "agentcore"


class TestGatewayIntegration:
    """Tests for gateway integration with MCP client."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_create_agent_with_memory_no_gateway(self):
        """Test creating agent when gateway is not configured."""
        from dova.runtime.agentcore_app import create_agent_with_memory

        # Without STACK_NAME, gateway should not be used
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("STACK_NAME", None)

            # Mock the necessary components - imports are inside the function
            with patch(
                "dova.config.providers.create_llm_router_from_settings"
            ) as mock_router:
                mock_router.return_value = MagicMock()

                orchestrator = await create_agent_with_memory(
                    user_id="test-user",
                    session_id="test-session",
                )

                # Should return an orchestrator even without gateway
                assert orchestrator is not None
