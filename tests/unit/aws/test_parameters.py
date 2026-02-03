"""Tests for parameters module (SSM and Secrets Manager)."""

from unittest.mock import MagicMock, patch

import pytest

from dova.aws.parameters import ParameterManager, ParameterSetupResult


class TestParameterSetupResult:
    """Tests for ParameterSetupResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = ParameterSetupResult(
            success=True,
            parameters_created=["/test/param1", "/test/param2"],
            secrets_created=["/test/secret1"],
        )

        assert result.success is True
        assert len(result.parameters_created) == 2
        assert len(result.secrets_created) == 1

    def test_failure_result(self):
        """Test failure result."""
        result = ParameterSetupResult(
            success=False,
            error="Access denied",
        )

        assert result.success is False
        assert result.error == "Access denied"


class TestParameterManager:
    """Tests for ParameterManager class."""

    @pytest.fixture
    def manager(self):
        """Create ParameterManager with mocked clients."""
        with patch("boto3.client"):
            return ParameterManager(region="us-east-1")

    @patch("boto3.client")
    def test_initialization(self, _mock_client):
        """Test ParameterManager initialization."""
        manager = ParameterManager(region="us-west-2")

        assert manager.region == "us-west-2"

    @patch("boto3.client")
    def test_store_configuration_success(self, mock_client):
        """Test successful configuration storage."""
        mock_ssm = MagicMock()
        mock_secrets = MagicMock()

        def client_factory(service, **_kwargs):
            if service == "ssm":
                return mock_ssm
            return mock_secrets

        mock_client.side_effect = client_factory

        manager = ParameterManager()
        result = manager.store_configuration(
            stack_name="test-stack",
            cognito_domain="test.auth.us-east-1.amazoncognito.com",
            client_id="client123",
            client_secret="secret123",
        )

        assert result.success is True
        assert len(result.parameters_created) >= 2
        assert len(result.secrets_created) == 1

    @patch("boto3.client")
    def test_get_parameter(self, mock_client):
        """Test getting a parameter."""
        mock_ssm = MagicMock()
        mock_ssm.get_parameter.return_value = {
            "Parameter": {"Value": "test-value"}
        }
        mock_client.return_value = mock_ssm

        manager = ParameterManager()
        value = manager.get_parameter("/test/param")

        assert value == "test-value"

    @patch("boto3.client")
    def test_get_parameter_not_found(self, mock_client):
        """Test getting a non-existent parameter."""
        from botocore.exceptions import ClientError

        mock_ssm = MagicMock()
        mock_ssm.get_parameter.side_effect = ClientError(
            {"Error": {"Code": "ParameterNotFound"}}, "GetParameter"
        )
        mock_client.return_value = mock_ssm

        manager = ParameterManager()
        value = manager.get_parameter("/test/nonexistent")

        assert value is None

    @patch("boto3.client")
    def test_get_secret(self, mock_client):
        """Test getting a secret."""
        mock_secrets = MagicMock()
        mock_secrets.get_secret_value.return_value = {
            "SecretString": "secret-value"
        }
        mock_client.return_value = mock_secrets

        manager = ParameterManager()
        value = manager.get_secret("/test/secret")

        assert value == "secret-value"

    @patch("boto3.client")
    def test_delete_configuration(self, mock_client):
        """Test deleting configuration."""
        mock_ssm = MagicMock()
        mock_secrets = MagicMock()

        def client_factory(service, **_kwargs):
            if service == "ssm":
                return mock_ssm
            return mock_secrets

        mock_client.side_effect = client_factory

        manager = ParameterManager()
        result = manager.delete_configuration("test-stack")

        assert result is True

    @patch("boto3.client")
    def test_validate_configuration(self, mock_client):
        """Test configuration validation."""
        mock_ssm = MagicMock()
        mock_ssm.get_parameter.return_value = {
            "Parameter": {"Value": "test-value"}
        }

        mock_secrets = MagicMock()
        mock_secrets.get_secret_value.return_value = {
            "SecretString": "secret-value"
        }

        def client_factory(service, **_kwargs):
            if service == "ssm":
                return mock_ssm
            return mock_secrets

        mock_client.side_effect = client_factory

        manager = ParameterManager()
        result = manager.validate_configuration("test-stack")

        assert result["cognito_provider"] is True
        assert result["machine_client_id"] is True
        assert result["machine_client_secret"] is True

    @patch("boto3.client")
    def test_generate_env_file(self, mock_client):
        """Test environment file generation."""
        import tempfile

        mock_ssm = MagicMock()
        mock_ssm.get_parameter.return_value = {
            "Parameter": {"Value": "test-value"}
        }

        mock_secrets = MagicMock()
        mock_secrets.get_secret_value.return_value = {
            "SecretString": "secret-value"
        }

        def client_factory(service, **_kwargs):
            if service == "ssm":
                return mock_ssm
            return mock_secrets

        mock_client.side_effect = client_factory

        manager = ParameterManager()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            result = manager.generate_env_file("test-stack", f.name)

            assert result is True

            # Read the file and check contents
            with open(f.name) as rf:
                content = rf.read()
                assert "STACK_NAME=test-stack" in content
