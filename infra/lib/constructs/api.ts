/**
 * API Construct - API Gateway and Lambda
 */

import * as cdk from 'aws-cdk-lib';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as cognito from 'aws-cdk-lib/aws-cognito';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as logs from 'aws-cdk-lib/aws-logs';
import { Construct } from 'constructs';

export interface ApiConstructProps {
  environment: string;
  userPool: cognito.UserPool;
  userPoolClient: cognito.UserPoolClient;
  profileTable: dynamodb.Table;
  cacheBucket: s3.Bucket;
  enableWaf?: boolean;
  bedrockModelId?: string;
}

export class ApiConstruct extends Construct {
  public readonly api: apigateway.RestApi;
  public readonly researchFunction: lambda.Function;

  constructor(scope: Construct, id: string, props: ApiConstructProps) {
    super(scope, id);

    // Lambda execution role with Bedrock access
    const lambdaRole = new iam.Role(this, 'LambdaRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          'service-role/AWSLambdaBasicExecutionRole'
        ),
      ],
    });

    // Bedrock permissions
    lambdaRole.addToPolicy(
      new iam.PolicyStatement({
        actions: [
          'bedrock:InvokeModel',
          'bedrock:InvokeModelWithResponseStream',
        ],
        resources: ['*'],
      })
    );

    // DynamoDB permissions
    props.profileTable.grantReadWriteData(lambdaRole);

    // S3 permissions
    props.cacheBucket.grantReadWrite(lambdaRole);

    // Research Lambda Function
    this.researchFunction = new lambda.Function(this, 'ResearchFunction', {
      functionName: `dova-${props.environment}-research`,
      runtime: lambda.Runtime.PYTHON_3_11,
      handler: 'index.handler',
      code: lambda.Code.fromAsset('../lambdas/gateway-tools'),
      role: lambdaRole,
      timeout: cdk.Duration.minutes(5),
      memorySize: 1024,
      environment: {
        ENVIRONMENT: props.environment,
        BEDROCK_MODEL_ID:
          props.bedrockModelId || 'anthropic.claude-sonnet-4-20250514-v1:0',
        PROFILE_TABLE_NAME: props.profileTable.tableName,
        CACHE_BUCKET_NAME: props.cacheBucket.bucketName,
      },
      logRetention: logs.RetentionDays.ONE_WEEK,
    });

    // API Gateway
    this.api = new apigateway.RestApi(this, 'DovaApi', {
      restApiName: `dova-${props.environment}-api`,
      description: 'DOVA Research Platform API',
      deployOptions: {
        stageName: props.environment,
        tracingEnabled: true,
        loggingLevel: apigateway.MethodLoggingLevel.INFO,
        dataTraceEnabled: props.environment !== 'production',
        metricsEnabled: true,
      },
      defaultCorsPreflightOptions: {
        allowOrigins: apigateway.Cors.ALL_ORIGINS,
        allowMethods: apigateway.Cors.ALL_METHODS,
        allowHeaders: [
          'Content-Type',
          'Authorization',
          'X-Api-Key',
          'X-Request-ID',
        ],
      },
    });

    // Cognito Authorizer
    const authorizer = new apigateway.CognitoUserPoolsAuthorizer(
      this,
      'CognitoAuthorizer',
      {
        cognitoUserPools: [props.userPool],
        authorizerName: 'dova-cognito-authorizer',
      }
    );

    // API Key for programmatic access
    const apiKey = this.api.addApiKey('DovaApiKey', {
      apiKeyName: `dova-${props.environment}-api-key`,
      description: 'API key for programmatic access to DOVA API',
    });

    const usagePlan = this.api.addUsagePlan('DovaUsagePlan', {
      name: `dova-${props.environment}-usage-plan`,
      throttle: {
        rateLimit: 100,
        burstLimit: 200,
      },
      quota: {
        limit: 10000,
        period: apigateway.Period.DAY,
      },
    });
    usagePlan.addApiKey(apiKey);
    usagePlan.addApiStage({ stage: this.api.deploymentStage });

    // Lambda Integration
    const researchIntegration = new apigateway.LambdaIntegration(
      this.researchFunction,
      {
        requestTemplates: {
          'application/json': '{ "statusCode": "200" }',
        },
      }
    );

    // API Routes
    const apiV1 = this.api.root.addResource('api').addResource('v1');

    // Health endpoint (no auth)
    const health = this.api.root.addResource('health');
    health.addMethod(
      'GET',
      new apigateway.MockIntegration({
        integrationResponses: [
          {
            statusCode: '200',
            responseTemplates: {
              'application/json':
                '{"status": "healthy", "service": "dova-api"}',
            },
          },
        ],
        passthroughBehavior: apigateway.PassthroughBehavior.NEVER,
        requestTemplates: {
          'application/json': '{"statusCode": 200}',
        },
      }),
      {
        methodResponses: [{ statusCode: '200' }],
      }
    );

    // Research endpoints (require auth)
    const research = apiV1.addResource('research');
    research.addMethod('POST', researchIntegration, {
      authorizer,
      authorizationType: apigateway.AuthorizationType.COGNITO,
    });

    // Search endpoints
    const search = apiV1.addResource('search');
    const searchSource = search.addResource('{source}');
    searchSource.addMethod('POST', researchIntegration, {
      authorizer,
      authorizationType: apigateway.AuthorizationType.COGNITO,
    });

    // Profile endpoints
    const profile = apiV1.addResource('profile');
    profile.addMethod('GET', researchIntegration, {
      authorizer,
      authorizationType: apigateway.AuthorizationType.COGNITO,
    });
    profile.addMethod('PUT', researchIntegration, {
      authorizer,
      authorizationType: apigateway.AuthorizationType.COGNITO,
    });

    // Validation endpoints
    const validate = apiV1.addResource('validate');
    validate.addMethod('POST', researchIntegration, {
      authorizer,
      authorizationType: apigateway.AuthorizationType.COGNITO,
    });
  }
}
