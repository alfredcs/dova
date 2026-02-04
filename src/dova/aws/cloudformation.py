"""CloudFormation template builder for DOVA Lambda deployment.

Generates CloudFormation templates for Lambda function and API Gateway.
"""

from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CloudFormationConfig:
    """Configuration for CloudFormation template generation."""

    stack_name: str
    lambda_s3_bucket: str
    lambda_s3_key: str
    role_arn: str
    region: str = "us-east-1"
    lambda_memory: int = 1024
    lambda_timeout: int = 300
    cognito_user_pool_arn: str | None = None
    enable_cors: bool = True


class CloudFormationBuilder:
    """Builds CloudFormation templates for DOVA Lambda deployment."""

    def __init__(self) -> None:
        self._logger = logger.bind(component="cloudformation_builder")

    def build_template(self, config: CloudFormationConfig) -> dict:
        """Build a CloudFormation template for Lambda + API Gateway.

        Args:
            config: Configuration for the deployment

        Returns:
            CloudFormation template as dict
        """
        self._logger.info("building_cloudformation_template", stack=config.stack_name)

        template = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": f"DOVA Lambda deployment for {config.stack_name}",
            "Parameters": {},
            "Resources": {},
            "Outputs": {},
        }

        # Add Lambda function
        self._add_lambda_function(template, config)

        # Add API Gateway
        self._add_api_gateway(template, config)

        # Add Lambda permission for API Gateway
        self._add_lambda_permission(template, config)

        # Add outputs
        self._add_outputs(template, config)

        return template

    def _add_lambda_function(self, template: dict, config: CloudFormationConfig) -> None:
        """Add Lambda function resource to template."""
        template["Resources"]["DovaLambdaFunction"] = {
            "Type": "AWS::Lambda::Function",
            "Properties": {
                "FunctionName": f"{config.stack_name}-dova-handler",
                "Description": "DOVA AgentCore Lambda handler",
                "Runtime": "python3.11",
                "Handler": "lambda_function.handler",
                "Code": {
                    "S3Bucket": config.lambda_s3_bucket,
                    "S3Key": config.lambda_s3_key,
                },
                "Role": config.role_arn,
                "MemorySize": config.lambda_memory,
                "Timeout": config.lambda_timeout,
                "Environment": {
                    "Variables": {
                        "STACK_NAME": config.stack_name,
                        "AWS_REGION_NAME": config.region,
                        "RUNTIME_MODE": "lambda",
                    }
                },
                "Tags": [
                    {"Key": "ManagedBy", "Value": "dova"},
                    {"Key": "Stack", "Value": config.stack_name},
                ],
            },
        }

    def _add_api_gateway(self, template: dict, config: CloudFormationConfig) -> None:
        """Add API Gateway resources to template."""
        # REST API
        template["Resources"]["DovaApi"] = {
            "Type": "AWS::ApiGateway::RestApi",
            "Properties": {
                "Name": f"{config.stack_name}-dova-api",
                "Description": "DOVA AgentCore API Gateway",
                "EndpointConfiguration": {"Types": ["REGIONAL"]},
                "Tags": [
                    {"Key": "ManagedBy", "Value": "dova"},
                    {"Key": "Stack", "Value": config.stack_name},
                ],
            },
        }

        # Resource: /invocations
        template["Resources"]["DovaApiResource"] = {
            "Type": "AWS::ApiGateway::Resource",
            "Properties": {
                "RestApiId": {"Ref": "DovaApi"},
                "ParentId": {"Fn::GetAtt": ["DovaApi", "RootResourceId"]},
                "PathPart": "invocations",
            },
        }

        # Method: POST /invocations
        method_props = {
            "RestApiId": {"Ref": "DovaApi"},
            "ResourceId": {"Ref": "DovaApiResource"},
            "HttpMethod": "POST",
            "AuthorizationType": "NONE",
            "Integration": {
                "Type": "AWS_PROXY",
                "IntegrationHttpMethod": "POST",
                "Uri": {
                    "Fn::Sub": (
                        "arn:aws:apigateway:${AWS::Region}:lambda:path/2015-03-31/"
                        "functions/${DovaLambdaFunction.Arn}/invocations"
                    )
                },
            },
        }

        # Add Cognito authorizer if configured
        if config.cognito_user_pool_arn:
            template["Resources"]["DovaCognitoAuthorizer"] = {
                "Type": "AWS::ApiGateway::Authorizer",
                "Properties": {
                    "Name": f"{config.stack_name}-cognito-authorizer",
                    "Type": "COGNITO_USER_POOLS",
                    "RestApiId": {"Ref": "DovaApi"},
                    "IdentitySource": "method.request.header.Authorization",
                    "ProviderARNs": [config.cognito_user_pool_arn],
                },
            }
            method_props["AuthorizationType"] = "COGNITO_USER_POOLS"
            method_props["AuthorizerId"] = {"Ref": "DovaCognitoAuthorizer"}

        template["Resources"]["DovaApiMethod"] = {
            "Type": "AWS::ApiGateway::Method",
            "Properties": method_props,
        }

        # Deployment - create before CORS so _add_cors_options can modify DependsOn
        template["Resources"]["DovaApiDeployment"] = {
            "Type": "AWS::ApiGateway::Deployment",
            "DependsOn": ["DovaApiMethod"],
            "Properties": {
                "RestApiId": {"Ref": "DovaApi"},
            },
        }

        # Add OPTIONS method for CORS if enabled
        if config.enable_cors:
            self._add_cors_options(template, config)

        # Stage
        template["Resources"]["DovaApiStage"] = {
            "Type": "AWS::ApiGateway::Stage",
            "Properties": {
                "StageName": "prod",
                "RestApiId": {"Ref": "DovaApi"},
                "DeploymentId": {"Ref": "DovaApiDeployment"},
                "Description": "Production stage",
                "Tags": [
                    {"Key": "ManagedBy", "Value": "dova"},
                    {"Key": "Stack", "Value": config.stack_name},
                ],
            },
        }

    def _add_cors_options(
        self, template: dict, config: CloudFormationConfig  # noqa: ARG002
    ) -> None:
        """Add CORS OPTIONS method to API Gateway."""
        template["Resources"]["DovaApiOptionsMethod"] = {
            "Type": "AWS::ApiGateway::Method",
            "Properties": {
                "RestApiId": {"Ref": "DovaApi"},
                "ResourceId": {"Ref": "DovaApiResource"},
                "HttpMethod": "OPTIONS",
                "AuthorizationType": "NONE",
                "Integration": {
                    "Type": "MOCK",
                    "RequestTemplates": {"application/json": '{"statusCode": 200}'},
                    "IntegrationResponses": [
                        {
                            "StatusCode": "200",
                            "ResponseParameters": {
                                "method.response.header.Access-Control-Allow-Headers": (
                                    "'Content-Type,Authorization,X-Amz-Date,X-Api-Key'"
                                ),
                                "method.response.header.Access-Control-Allow-Methods": (
                                    "'POST,OPTIONS'"
                                ),
                                "method.response.header.Access-Control-Allow-Origin": "'*'",
                            },
                        }
                    ],
                },
                "MethodResponses": [
                    {
                        "StatusCode": "200",
                        "ResponseParameters": {
                            "method.response.header.Access-Control-Allow-Headers": True,
                            "method.response.header.Access-Control-Allow-Methods": True,
                            "method.response.header.Access-Control-Allow-Origin": True,
                        },
                    }
                ],
            },
        }

        # Update deployment to depend on OPTIONS method too
        template["Resources"]["DovaApiDeployment"]["DependsOn"].append(
            "DovaApiOptionsMethod"
        )

    def _add_lambda_permission(
        self, template: dict, config: CloudFormationConfig  # noqa: ARG002
    ) -> None:
        """Add Lambda permission for API Gateway invocation."""
        template["Resources"]["DovaLambdaApiPermission"] = {
            "Type": "AWS::Lambda::Permission",
            "Properties": {
                "FunctionName": {"Ref": "DovaLambdaFunction"},
                "Action": "lambda:InvokeFunction",
                "Principal": "apigateway.amazonaws.com",
                "SourceArn": {
                    "Fn::Sub": (
                        "arn:aws:execute-api:${AWS::Region}:${AWS::AccountId}:"
                        "${DovaApi}/*/POST/invocations"
                    )
                },
            },
        }

    def _add_outputs(self, template: dict, config: CloudFormationConfig) -> None:
        """Add CloudFormation outputs."""
        template["Outputs"]["LambdaFunctionArn"] = {
            "Description": "ARN of the DOVA Lambda function",
            "Value": {"Fn::GetAtt": ["DovaLambdaFunction", "Arn"]},
            "Export": {"Name": f"{config.stack_name}-lambda-arn"},
        }

        template["Outputs"]["ApiGatewayUrl"] = {
            "Description": "URL of the DOVA API Gateway endpoint",
            "Value": {
                "Fn::Sub": (
                    "https://${DovaApi}.execute-api.${AWS::Region}.amazonaws.com/prod/invocations"
                )
            },
            "Export": {"Name": f"{config.stack_name}-api-url"},
        }

        template["Outputs"]["ApiGatewayId"] = {
            "Description": "ID of the API Gateway",
            "Value": {"Ref": "DovaApi"},
            "Export": {"Name": f"{config.stack_name}-api-id"},
        }
