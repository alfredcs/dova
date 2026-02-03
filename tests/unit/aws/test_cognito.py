"""Tests for Cognito setup module."""

from unittest.mock import MagicMock, patch

import pytest

from dova.aws.cognito import CognitoManager, CognitoSetupResult


class TestCognitoSetupResult:
    """Tests for CognitoSetupResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = CognitoSetupResult(
            success=True,
            user_pool_id="us-east-1_ABC123",
            client_id="client123",
            client_secret="secret123",
            domain="test.auth.us-east-1.amazoncognito.com",
        )

        assert result.success is True
        assert result.user_pool_id == "us-east-1_ABC123"
        assert result.client_id == "client123"
        assert result.client_secret == "secret123"
        assert result.domain is not None

    def test_failure_result(self):
        """Test failure result."""
        result = CognitoSetupResult(
            success=False,
            error="User pool creation failed",
        )

        assert result.success is False
        assert result.error == "User pool creation failed"
        assert result.user_pool_id is None


class TestCognitoManager:
    """Tests for CognitoManager class."""

    @pytest.fixture
    def manager(self):
        """Create CognitoManager with mocked client."""
        with patch("boto3.client"):
            return CognitoManager(region="us-east-1")

    @patch("boto3.client")
    def test_initialization(self, mock_client):
        """Test CognitoManager initialization."""
        manager = CognitoManager(region="us-west-2")

        assert manager.region == "us-west-2"
        mock_client.assert_called_with("cognito-idp", region_name="us-west-2")

    @patch("boto3.client")
    def test_setup_cognito_success(self, mock_client):
        """Test successful Cognito setup."""
        mock_cognito = MagicMock()

        # Mock list_user_pools to return empty (no existing pool)
        mock_cognito.list_user_pools.return_value = {"UserPools": []}

        # Mock create_user_pool
        mock_cognito.create_user_pool.return_value = {
            "UserPool": {"Id": "us-east-1_ABC123"}
        }

        # Mock create_resource_server
        mock_cognito.create_resource_server.return_value = {}

        # Mock list_user_pool_clients to return empty (no existing client)
        mock_cognito.list_user_pool_clients.return_value = {"UserPoolClients": []}

        # Mock create_user_pool_client
        mock_cognito.create_user_pool_client.return_value = {
            "UserPoolClient": {
                "ClientId": "client123",
                "ClientSecret": "secret123",
            }
        }

        # Mock describe_user_pool for domain check
        mock_cognito.describe_user_pool.return_value = {
            "UserPool": {"Domain": None}
        }

        # Mock create_user_pool_domain
        mock_cognito.create_user_pool_domain.return_value = {}

        mock_client.return_value = mock_cognito

        manager = CognitoManager()
        result = manager.setup_cognito("test-stack")

        assert result.success is True
        assert result.user_pool_id == "us-east-1_ABC123"
        assert result.client_id == "client123"
        assert result.client_secret == "secret123"

    @patch("boto3.client")
    def test_setup_cognito_existing_pool(self, mock_client):
        """Test setup with existing user pool."""
        mock_cognito = MagicMock()

        # Mock list_user_pools to return existing pool
        mock_cognito.list_user_pools.return_value = {
            "UserPools": [
                {"Id": "us-east-1_EXISTING", "Name": "test-stack-dova-pool"}
            ]
        }

        # Mock list_user_pool_clients with existing client
        mock_cognito.list_user_pool_clients.return_value = {
            "UserPoolClients": [
                {"ClientId": "existing-client", "ClientName": "test-stack-machine-client"}
            ]
        }

        # Mock describe_user_pool_client
        mock_cognito.describe_user_pool_client.return_value = {
            "UserPoolClient": {
                "ClientId": "existing-client",
                "ClientSecret": "existing-secret",
            }
        }

        # Mock describe_user_pool for domain
        mock_cognito.describe_user_pool.return_value = {
            "UserPool": {"Domain": "test-domain"}
        }

        # Mock describe_resource_server (exists)
        mock_cognito.describe_resource_server.return_value = {}

        mock_client.return_value = mock_cognito

        manager = CognitoManager()
        result = manager.setup_cognito("test-stack")

        assert result.success is True
        assert result.user_pool_id == "us-east-1_EXISTING"
        assert result.client_id == "existing-client"

    @patch("boto3.client")
    def test_validate_setup(self, mock_client):
        """Test validation of Cognito setup."""
        mock_cognito = MagicMock()

        # Mock list_user_pools
        mock_cognito.list_user_pools.return_value = {
            "UserPools": [
                {"Id": "us-east-1_ABC123", "Name": "test-stack-dova-pool"}
            ]
        }

        # Mock list_user_pool_clients
        mock_cognito.list_user_pool_clients.return_value = {
            "UserPoolClients": [
                {"ClientId": "client123", "ClientName": "test-stack-machine-client"}
            ]
        }

        # Mock describe_resource_server
        mock_cognito.describe_resource_server.return_value = {}

        # Mock describe_user_pool
        mock_cognito.describe_user_pool.return_value = {
            "UserPool": {"Domain": "test-domain"}
        }

        mock_client.return_value = mock_cognito

        manager = CognitoManager()
        result = manager.validate_setup("test-stack")

        assert result["user_pool"] is True
        assert result["app_client"] is True
        assert result["resource_server"] is True
        assert result["domain"] is True

    @patch("boto3.client")
    def test_delete_cognito_resources(self, mock_client):
        """Test deleting Cognito resources."""
        mock_cognito = MagicMock()

        # Mock list_user_pools
        mock_cognito.list_user_pools.return_value = {
            "UserPools": [
                {"Id": "us-east-1_ABC123", "Name": "test-stack-dova-pool"}
            ]
        }

        # Mock describe_user_pool
        mock_cognito.describe_user_pool.return_value = {
            "UserPool": {"Domain": "test-domain"}
        }

        mock_client.return_value = mock_cognito

        manager = CognitoManager()
        result = manager.delete_cognito_resources("test-stack")

        assert result is True
        mock_cognito.delete_user_pool_domain.assert_called_once()
        mock_cognito.delete_user_pool.assert_called_once()
