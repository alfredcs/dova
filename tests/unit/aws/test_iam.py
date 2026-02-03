"""Tests for IAM setup module."""

from unittest.mock import MagicMock, patch

import pytest

from dova.aws.iam import (
    IAMManager,
    IAMSetupResult,
    get_agentcore_policy,
    get_bedrock_policy,
    get_required_setup_permissions,
    get_secrets_policy,
    get_ssm_policy,
)


class TestIAMPolicies:
    """Tests for IAM policy generation functions."""

    def test_get_bedrock_policy(self):
        """Test Bedrock policy generation."""
        policy = get_bedrock_policy("us-east-1", "123456789012")

        assert policy["Version"] == "2012-10-17"
        assert len(policy["Statement"]) == 2

        # Check model invocation statement
        invoke_stmt = policy["Statement"][0]
        assert "bedrock:InvokeModel" in invoke_stmt["Action"]
        assert "bedrock:InvokeModelWithResponseStream" in invoke_stmt["Action"]

    def test_get_agentcore_policy(self):
        """Test AgentCore policy generation."""
        policy = get_agentcore_policy("us-east-1", "123456789012", "test-stack")

        assert policy["Version"] == "2012-10-17"
        assert len(policy["Statement"]) == 2

        # Check memory statement
        memory_stmt = policy["Statement"][0]
        assert "bedrock:CreateMemory" in memory_stmt["Action"]
        assert "bedrock:InvokeMemory" in memory_stmt["Action"]

    def test_get_ssm_policy(self):
        """Test SSM policy generation."""
        policy = get_ssm_policy("us-east-1", "123456789012", "test-stack")

        assert policy["Version"] == "2012-10-17"
        stmt = policy["Statement"][0]
        assert "ssm:GetParameter" in stmt["Action"]
        assert "ssm:PutParameter" in stmt["Action"]
        assert "test-stack" in stmt["Resource"]

    def test_get_secrets_policy(self):
        """Test Secrets Manager policy generation."""
        policy = get_secrets_policy("us-east-1", "123456789012", "test-stack")

        assert policy["Version"] == "2012-10-17"
        stmt = policy["Statement"][0]
        assert "secretsmanager:GetSecretValue" in stmt["Action"]
        assert "secretsmanager:CreateSecret" in stmt["Action"]


class TestGetRequiredSetupPermissions:
    """Tests for required permissions list."""

    def test_returns_list(self):
        """Test that permissions list is returned."""
        perms = get_required_setup_permissions()

        assert isinstance(perms, list)
        assert len(perms) > 0

    def test_includes_iam_permissions(self):
        """Test IAM permissions are included."""
        perms = get_required_setup_permissions()

        assert "iam:CreateRole" in perms
        assert "iam:CreatePolicy" in perms
        assert "iam:AttachRolePolicy" in perms

    def test_includes_cognito_permissions(self):
        """Test Cognito permissions are included."""
        perms = get_required_setup_permissions()

        assert "cognito-idp:CreateUserPool" in perms
        assert "cognito-idp:CreateUserPoolClient" in perms

    def test_includes_ssm_permissions(self):
        """Test SSM permissions are included."""
        perms = get_required_setup_permissions()

        assert "ssm:PutParameter" in perms
        assert "ssm:GetParameter" in perms

    def test_includes_secrets_permissions(self):
        """Test Secrets Manager permissions are included."""
        perms = get_required_setup_permissions()

        assert "secretsmanager:CreateSecret" in perms
        assert "secretsmanager:GetSecretValue" in perms


class TestIAMManager:
    """Tests for IAMManager class."""

    @pytest.fixture
    def manager(self):
        """Create IAMManager with mocked clients."""
        with patch("boto3.client"):
            return IAMManager(region="us-east-1")

    @patch("boto3.client")
    def test_initialization(self, mock_client):
        """Test IAMManager initialization."""
        manager = IAMManager(region="us-west-2")

        assert manager.region == "us-west-2"
        mock_client.assert_called()

    @patch("boto3.client")
    def test_get_account_id(self, mock_client):
        """Test getting AWS account ID."""
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_client.return_value = mock_sts

        manager = IAMManager()
        account_id = manager.get_account_id()

        assert account_id == "123456789012"

    @patch("boto3.client")
    def test_create_dova_role_success(self, mock_client):
        """Test successful role creation."""
        mock_iam = MagicMock()
        mock_iam.create_role.return_value = {}
        mock_iam.create_policy.return_value = {
            "Policy": {"Arn": "arn:aws:iam::123456789012:policy/test-policy"}
        }
        mock_iam.list_policy_versions.return_value = {"Versions": []}

        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

        def client_factory(service, **_kwargs):
            if service == "sts":
                return mock_sts
            return mock_iam

        mock_client.side_effect = client_factory

        manager = IAMManager()
        result = manager.create_dova_role("test-stack")

        assert result.success is True
        assert result.role_arn is not None


class TestIAMSetupResult:
    """Tests for IAMSetupResult dataclass."""

    def test_success_result(self):
        """Test successful result."""
        result = IAMSetupResult(
            success=True,
            role_arn="arn:aws:iam::123456789012:role/test-role",
            policy_arns=["arn:aws:iam::123456789012:policy/test-policy"],
        )

        assert result.success is True
        assert result.role_arn is not None
        assert len(result.policy_arns) == 1

    def test_failure_result(self):
        """Test failure result."""
        result = IAMSetupResult(
            success=False,
            error="Access denied",
            missing_permissions=["iam:CreateRole"],
        )

        assert result.success is False
        assert result.error == "Access denied"
        assert "iam:CreateRole" in result.missing_permissions
