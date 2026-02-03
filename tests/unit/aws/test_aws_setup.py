"""Tests for AWS setup module."""

from unittest.mock import MagicMock, patch

import pytest

from dova.aws.setup import AWSSetup, SetupConfig, SetupPhase, SetupResult


class TestSetupConfig:
    """Tests for SetupConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SetupConfig(stack_name="test-stack")

        assert config.stack_name == "test-stack"
        assert config.region == "us-east-1"
        assert config.include_bedrock is True
        assert config.include_agentcore is True
        assert config.generate_env_file is True
        assert config.env_file_path == ".env.aws"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SetupConfig(
            stack_name="custom-stack",
            region="us-west-2",
            include_bedrock=False,
            gateway_url="https://gateway.example.com",
        )

        assert config.stack_name == "custom-stack"
        assert config.region == "us-west-2"
        assert config.include_bedrock is False
        assert config.gateway_url == "https://gateway.example.com"


class TestAWSSetup:
    """Tests for AWSSetup class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SetupConfig(
            stack_name="test-stack",
            region="us-east-1",
        )

    @pytest.fixture
    def setup(self, config):
        """Create AWSSetup instance with mocked managers."""
        with patch("dova.aws.setup.AWSValidator"), patch(
            "dova.aws.setup.IAMManager"
        ), patch("dova.aws.setup.CognitoManager"), patch(
            "dova.aws.setup.ParameterManager"
        ):
            return AWSSetup(config)

    def test_initialization(self, config):
        """Test AWSSetup initialization."""
        with patch("dova.aws.setup.AWSValidator"), patch(
            "dova.aws.setup.IAMManager"
        ), patch("dova.aws.setup.CognitoManager"), patch(
            "dova.aws.setup.ParameterManager"
        ):
            setup = AWSSetup(config)

            assert setup.config.stack_name == "test-stack"
            assert setup.config.region == "us-east-1"

    @patch("dova.aws.setup.AWSValidator")
    @patch("dova.aws.setup.IAMManager")
    @patch("dova.aws.setup.CognitoManager")
    @patch("dova.aws.setup.ParameterManager")
    def test_run_success(
        self,
        mock_params,
        mock_cognito,
        mock_iam,
        mock_validator,
        config,
    ):
        """Test successful setup run."""
        # Setup mocks
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate_credentials.return_value = MagicMock(
            valid=True, account_id="123456789012"
        )
        mock_validator_instance.validate_iam_permissions.return_value = MagicMock(
            valid=True
        )
        mock_validator_instance.validate_bedrock_access.return_value = MagicMock(
            valid=True, errors=[]
        )
        mock_validator_instance.validate_complete_setup.return_value = MagicMock(
            valid=True
        )
        mock_validator.return_value = mock_validator_instance

        mock_iam_instance = MagicMock()
        mock_iam_instance.create_dova_role.return_value = MagicMock(
            success=True, role_arn="arn:aws:iam::123456789012:role/test-role"
        )
        mock_iam.return_value = mock_iam_instance

        mock_cognito_instance = MagicMock()
        mock_cognito_instance.setup_cognito.return_value = MagicMock(
            success=True,
            user_pool_id="us-east-1_ABC123",
            client_id="client123",
            client_secret="secret123",
            domain="test.auth.us-east-1.amazoncognito.com",
        )
        mock_cognito.return_value = mock_cognito_instance

        mock_params_instance = MagicMock()
        mock_params_instance.store_configuration.return_value = MagicMock(
            success=True
        )
        mock_params_instance.generate_env_file.return_value = True
        mock_params.return_value = mock_params_instance

        setup = AWSSetup(config)
        result = setup.run()

        assert result.success is True
        assert result.phase == SetupPhase.FINALIZE

    @patch("dova.aws.setup.AWSValidator")
    @patch("dova.aws.setup.IAMManager")
    @patch("dova.aws.setup.CognitoManager")
    @patch("dova.aws.setup.ParameterManager")
    def test_run_credential_failure(
        self,
        mock_params,  # noqa: ARG002
        mock_cognito,  # noqa: ARG002
        mock_iam,  # noqa: ARG002
        mock_validator,
        config,
    ):
        """Test setup failure due to credential issues."""
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate_credentials.return_value = MagicMock(
            valid=False, errors=["No credentials found"]
        )
        mock_validator.return_value = mock_validator_instance

        setup = AWSSetup(config)
        result = setup.run()

        assert result.success is False
        assert result.phase == SetupPhase.VALIDATE_CREDENTIALS
        assert len(result.errors) > 0

    @patch("dova.aws.setup.AWSValidator")
    @patch("dova.aws.setup.IAMManager")
    @patch("dova.aws.setup.CognitoManager")
    @patch("dova.aws.setup.ParameterManager")
    def test_teardown(
        self,
        mock_params,
        mock_cognito,
        mock_iam,
        mock_validator,  # noqa: ARG002
        config,
    ):
        """Test teardown functionality."""
        mock_params_instance = MagicMock()
        mock_params_instance.delete_configuration.return_value = True
        mock_params.return_value = mock_params_instance

        mock_cognito_instance = MagicMock()
        mock_cognito_instance.delete_cognito_resources.return_value = True
        mock_cognito.return_value = mock_cognito_instance

        mock_iam_instance = MagicMock()
        mock_iam_instance.delete_dova_resources.return_value = True
        mock_iam.return_value = mock_iam_instance

        setup = AWSSetup(config)
        result = setup.teardown()

        assert result.success is True
        mock_params_instance.delete_configuration.assert_called_once()
        mock_cognito_instance.delete_cognito_resources.assert_called_once()
        mock_iam_instance.delete_dova_resources.assert_called_once()


class TestSetupResult:
    """Tests for SetupResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = SetupResult(success=False)

        assert result.success is False
        assert result.phase is None
        assert result.errors == []
        assert result.warnings == []

    def test_with_phase(self):
        """Test result with phase."""
        result = SetupResult(
            success=True,
            phase=SetupPhase.FINALIZE,
        )

        assert result.success is True
        assert result.phase == SetupPhase.FINALIZE
