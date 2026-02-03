"""Tests for AWS validators module."""

from unittest.mock import MagicMock, patch

import pytest

from dova.aws.validators import (
    AWSValidator,
    ValidationResult,
    format_validation_result,
)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = ValidationResult(valid=False)

        assert result.valid is False
        assert result.checks == {}
        assert result.errors == []
        assert result.warnings == []

    def test_with_values(self):
        """Test with custom values."""
        result = ValidationResult(
            valid=True,
            checks={"credentials": True, "bedrock": True},
            account_id="123456789012",
            region="us-east-1",
        )

        assert result.valid is True
        assert result.checks["credentials"] is True
        assert result.account_id == "123456789012"


class TestFormatValidationResult:
    """Tests for format_validation_result function."""

    def test_format_passed(self):
        """Test formatting passed result."""
        result = ValidationResult(
            valid=True,
            checks={"credentials": True, "bedrock": True},
            account_id="123456789012",
            region="us-east-1",
        )

        output = format_validation_result(result)

        assert "PASSED" in output
        assert "123456789012" in output
        assert "[OK]" in output

    def test_format_failed(self):
        """Test formatting failed result."""
        result = ValidationResult(
            valid=False,
            checks={"credentials": False},
            errors=["No credentials found"],
            region="us-east-1",
        )

        output = format_validation_result(result)

        assert "FAILED" in output
        assert "[FAIL]" in output
        assert "No credentials found" in output

    def test_format_with_warnings(self):
        """Test formatting with warnings."""
        result = ValidationResult(
            valid=True,
            checks={"credentials": True},
            warnings=["Some model not available"],
        )

        output = format_validation_result(result)

        assert "Warnings:" in output
        assert "Some model not available" in output


class TestAWSValidator:
    """Tests for AWSValidator class."""

    @pytest.fixture
    def validator(self):
        """Create AWSValidator instance."""
        return AWSValidator(region="us-east-1")

    @patch("boto3.client")
    def test_validate_credentials_success(self, mock_client, validator):
        """Test successful credential validation."""
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {
            "Account": "123456789012",
            "Arn": "arn:aws:iam::123456789012:user/test",
        }
        mock_client.return_value = mock_sts

        result = validator.validate_credentials()

        assert result.valid is True
        assert result.account_id == "123456789012"
        assert result.checks["credentials"] is True

    @patch("boto3.client")
    def test_validate_credentials_no_creds(self, mock_client, validator):
        """Test credential validation with no credentials."""
        from botocore.exceptions import NoCredentialsError

        mock_client.side_effect = NoCredentialsError()

        result = validator.validate_credentials()

        assert result.valid is False
        assert len(result.errors) > 0
        assert "credentials" in result.errors[0].lower()

    @patch("boto3.client")
    def test_validate_bedrock_access(self, mock_client, validator):
        """Test Bedrock access validation."""
        mock_bedrock = MagicMock()
        mock_bedrock.list_foundation_models.return_value = {
            "modelSummaries": [
                {"modelId": "anthropic.claude-3-sonnet-20240229-v1:0"},
                {"modelId": "anthropic.claude-3-haiku-20240307-v1:0"},
            ]
        }

        mock_bedrock_runtime = MagicMock()

        def client_factory(service, **_kwargs):
            if service == "bedrock-runtime":
                return mock_bedrock_runtime
            return mock_bedrock

        mock_client.side_effect = client_factory

        result = validator.validate_bedrock_access()

        assert result.checks["bedrock_list_models"] is True

    @patch("boto3.client")
    def test_validate_iam_permissions(self, mock_client, validator):
        """Test IAM permission validation."""
        mock_iam = MagicMock()
        mock_iam.list_policies.return_value = {"Policies": []}

        mock_cognito = MagicMock()
        mock_cognito.list_user_pools.return_value = {"UserPools": []}

        mock_ssm = MagicMock()
        mock_ssm.describe_parameters.return_value = {"Parameters": []}

        mock_secrets = MagicMock()
        mock_secrets.list_secrets.return_value = {"SecretList": []}

        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

        def client_factory(service, **_kwargs):
            clients = {
                "iam": mock_iam,
                "cognito-idp": mock_cognito,
                "ssm": mock_ssm,
                "secretsmanager": mock_secrets,
                "sts": mock_sts,
            }
            return clients.get(service, MagicMock())

        mock_client.side_effect = client_factory

        result = validator.validate_iam_permissions(["iam:CreateRole", "ssm:PutParameter"])

        assert result.valid is True

    @patch("boto3.client")
    def test_validate_complete_setup_credential_failure(self, mock_client, validator):
        """Test complete setup validation fails when credentials are invalid."""
        from botocore.exceptions import NoCredentialsError

        mock_client.side_effect = NoCredentialsError()

        result = validator.validate_complete_setup("test-stack")

        # Should fail at credentials check
        assert result.valid is False
        assert result.checks.get("credentials") is False
