"""Tests for gateway MCP client."""

import os
from unittest.mock import MagicMock, patch

import pytest

from dova.config.mcp_servers import MCPServerConfig, MCPTransport


class TestCreateGatewayMcpClient:
    """Tests for create_gateway_mcp_client function."""

    def test_returns_none_when_stack_name_not_set(self):
        """Test that None is returned when STACK_NAME is not set."""
        from dova.tools.mcp_registry import create_gateway_mcp_client

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("STACK_NAME", None)
            result = create_gateway_mcp_client()

        assert result is None

    @patch("dova.services.gateway_auth.get_gateway_url")
    @patch("dova.services.gateway_auth.get_gateway_access_token")
    def test_returns_config_when_configured(
        self,
        mock_get_token,
        mock_get_url,
    ):
        """Test that MCPServerConfig is returned when properly configured."""
        from dova.tools.mcp_registry import create_gateway_mcp_client

        mock_get_token.return_value = "test-token"
        mock_get_url.return_value = "https://gateway.example.com/mcp"

        with patch.dict(os.environ, {"STACK_NAME": "test-stack"}):
            result = create_gateway_mcp_client()

        assert result is not None
        assert isinstance(result, MCPServerConfig)
        assert result.name == "gateway"
        assert result.transport == MCPTransport.STREAMABLE_HTTP
        assert result.url == "https://gateway.example.com/mcp"
        assert result.headers["Authorization"] == "Bearer test-token"
        assert result.priority == 0

    @patch("dova.services.gateway_auth.get_gateway_url")
    @patch("dova.services.gateway_auth.get_gateway_access_token")
    def test_uses_provided_token(
        self,
        mock_get_token,
        mock_get_url,
    ):
        """Test that provided access token is used."""
        from dova.tools.mcp_registry import create_gateway_mcp_client

        mock_get_url.return_value = "https://gateway.example.com/mcp"

        with patch.dict(os.environ, {"STACK_NAME": "test-stack"}):
            result = create_gateway_mcp_client(access_token="provided-token")

        assert result is not None
        assert result.headers["Authorization"] == "Bearer provided-token"
        mock_get_token.assert_not_called()

    @patch("dova.services.gateway_auth.get_gateway_access_token")
    def test_returns_none_on_auth_error(self, mock_get_token):
        """Test that None is returned on authentication error."""
        from dova.tools.mcp_registry import create_gateway_mcp_client

        mock_get_token.side_effect = Exception("Auth failed")

        with patch.dict(os.environ, {"STACK_NAME": "test-stack"}):
            result = create_gateway_mcp_client()

        assert result is None


class TestMCPClientStreamableHttp:
    """Tests for MCPClient streamable HTTP transport."""

    @pytest.fixture
    def mcp_client(self):
        """Create an MCPClient instance for testing."""
        from dova.tools.mcp_registry import MCPClient

        return MCPClient()

    def test_streamable_http_transport_in_invoke_internal(self, mcp_client):
        """Test that STREAMABLE_HTTP transport is handled."""
        # Verify the transport type is recognized in the mapping
        _ = MCPServerConfig(
            name="test-gateway",
            description="Test gateway",
            transport=MCPTransport.STREAMABLE_HTTP,
            url="https://test.gateway.com/mcp",
        )

        # The method should exist on the client
        assert hasattr(mcp_client, "_invoke_streamable_http")

    @pytest.mark.asyncio
    async def test_invoke_streamable_http_requires_url(self, mcp_client):
        """Test that streamable HTTP requires URL."""
        server_config = MCPServerConfig(
            name="test-gateway",
            description="Test gateway",
            transport=MCPTransport.STREAMABLE_HTTP,
            url=None,  # No URL
        )

        with pytest.raises(ValueError, match="No URL configured"):
            await mcp_client._invoke_streamable_http(
                server_config, "test_tool", {}
            )

    def test_parse_streamable_result_text_content(self, mcp_client):
        """Test parsing streamable result with text content."""
        # Create mock result with content
        mock_result = MagicMock()
        mock_item = MagicMock()
        mock_item.text = '{"key": "value"}'
        mock_result.content = [mock_item]

        result = mcp_client._parse_streamable_result(mock_result)

        assert result == {"key": "value"}

    def test_parse_streamable_result_plain_text(self, mcp_client):
        """Test parsing streamable result with plain text."""
        mock_result = MagicMock()
        mock_item = MagicMock()
        mock_item.text = "Just some text"
        mock_result.content = [mock_item]

        result = mcp_client._parse_streamable_result(mock_result)

        assert result == "Just some text"

    def test_parse_streamable_result_no_content(self, mcp_client):
        """Test parsing streamable result without content."""
        mock_result = MagicMock()
        mock_result.content = None

        result = mcp_client._parse_streamable_result(mock_result)

        assert result is None


class TestMCPTransportEnum:
    """Tests for MCPTransport enum."""

    def test_streamable_http_exists(self):
        """Test that STREAMABLE_HTTP transport type exists."""
        assert hasattr(MCPTransport, "STREAMABLE_HTTP")
        assert MCPTransport.STREAMABLE_HTTP.value == "streamable_http"

    def test_all_transports(self):
        """Test all transport types are defined."""
        transports = list(MCPTransport)
        assert len(transports) == 4
        assert MCPTransport.STDIO in transports
        assert MCPTransport.SSE in transports
        assert MCPTransport.HTTP in transports
        assert MCPTransport.STREAMABLE_HTTP in transports
