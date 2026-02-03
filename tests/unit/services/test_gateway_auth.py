"""Tests for gateway authentication."""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestGetStackName:
    """Tests for get_stack_name function."""

    def test_get_stack_name_from_env(self):
        """Test getting stack name from environment variable."""
        from dova.services.gateway_auth import get_stack_name

        with patch.dict(os.environ, {"STACK_NAME": "my-test-stack"}):
            assert get_stack_name() == "my-test-stack"

    def test_get_stack_name_missing(self):
        """Test error when STACK_NAME not set."""
        from dova.services.gateway_auth import get_stack_name

        with patch.dict(os.environ, {}, clear=True):
            # Remove STACK_NAME if it exists
            os.environ.pop("STACK_NAME", None)
            with pytest.raises(ValueError, match="STACK_NAME"):
                get_stack_name()


class TestGatewayAccessToken:
    """Tests for get_gateway_access_token function."""

    @patch("dova.services.gateway_auth.get_secret")
    @patch("dova.services.gateway_auth.get_ssm_parameter")
    @patch("httpx.Client")
    def test_get_gateway_access_token_success(
        self,
        mock_client_class,
        mock_ssm,
        mock_secret,
    ):
        """Test successful token acquisition."""
        from dova.services.gateway_auth import (
            clear_token_cache,
            get_gateway_access_token,
        )

        # Clear cache first
        clear_token_cache()

        # Setup mocks
        mock_ssm.side_effect = lambda name: {
            "/test-stack/cognito_provider": "test.auth.region.amazoncognito.com",
            "/test-stack/machine_client_id": "test-client-id",
        }.get(name, "")

        mock_secret.return_value = "test-client-secret"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "test-token-123",
            "token_type": "Bearer",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=None)
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        with patch.dict(os.environ, {"STACK_NAME": "test-stack"}):
            token = get_gateway_access_token()

        assert token == "test-token-123"

    @patch("dova.services.gateway_auth.get_secret")
    @patch("dova.services.gateway_auth.get_ssm_parameter")
    def test_get_gateway_access_token_cached(
        self,
        mock_ssm,
        mock_secret,
    ):
        """Test that cached token is returned."""
        import time

        from dova.services.gateway_auth import (
            GatewayToken,
            _token_cache,
            clear_token_cache,
            get_gateway_access_token,
        )

        clear_token_cache()

        # Pre-populate cache
        _token_cache["gateway_token:test-stack"] = GatewayToken(
            access_token="cached-token",
            token_type="Bearer",
            expires_in=3600,
            expires_at=time.time() + 3600,
        )

        with patch.dict(os.environ, {"STACK_NAME": "test-stack"}):
            token = get_gateway_access_token()

        # Should return cached token without making API calls
        assert token == "cached-token"
        mock_ssm.assert_not_called()
        mock_secret.assert_not_called()

    def test_clear_token_cache(self):
        """Test clearing token cache."""
        import time

        from dova.services.gateway_auth import (
            GatewayToken,
            _token_cache,
            clear_token_cache,
        )

        # Add something to cache
        _token_cache["test"] = GatewayToken(
            access_token="test",
            token_type="Bearer",
            expires_in=3600,
            expires_at=time.time() + 3600,
        )

        clear_token_cache()

        assert len(_token_cache) == 0


class TestAwsConfig:
    """Tests for AWS config utilities."""

    @patch("boto3.client")
    def test_get_ssm_parameter(self, mock_boto_client):
        """Test SSM parameter retrieval."""
        from dova.services.aws_config import clear_ssm_cache, get_ssm_parameter

        clear_ssm_cache()

        mock_ssm = MagicMock()
        mock_ssm.get_parameter.return_value = {
            "Parameter": {"Value": "test-value"}
        }
        mock_boto_client.return_value = mock_ssm

        result = get_ssm_parameter("/test/param")

        assert result == "test-value"
        mock_ssm.get_parameter.assert_called_once_with(
            Name="/test/param", WithDecryption=True
        )

    @patch("boto3.client")
    def test_get_secret(self, mock_boto_client):
        """Test Secrets Manager retrieval."""
        from dova.services.aws_config import get_secret

        mock_sm = MagicMock()
        mock_sm.get_secret_value.return_value = {
            "SecretString": "secret-value"
        }
        mock_boto_client.return_value = mock_sm

        result = get_secret("test-secret")

        assert result == "secret-value"
        mock_sm.get_secret_value.assert_called_once_with(SecretId="test-secret")

    @patch("boto3.client")
    def test_get_secret_json(self, mock_boto_client):
        """Test JSON secret retrieval and parsing."""
        from dova.services.aws_config import get_secret_json

        mock_sm = MagicMock()
        mock_sm.get_secret_value.return_value = {
            "SecretString": '{"key": "value", "number": 42}'
        }
        mock_boto_client.return_value = mock_sm

        result = get_secret_json("test-secret")

        assert result == {"key": "value", "number": 42}

    @patch("boto3.client")
    def test_get_ssm_parameters_by_path(self, mock_boto_client):
        """Test batch parameter retrieval by path."""
        from dova.services.aws_config import get_ssm_parameters_by_path

        mock_ssm = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Parameters": [
                    {"Name": "/mystack/param1", "Value": "value1"},
                    {"Name": "/mystack/param2", "Value": "value2"},
                ]
            }
        ]
        mock_ssm.get_paginator.return_value = mock_paginator
        mock_boto_client.return_value = mock_ssm

        result = get_ssm_parameters_by_path("/mystack/")

        assert result == {
            "/mystack/param1": "value1",
            "/mystack/param2": "value2",
        }
