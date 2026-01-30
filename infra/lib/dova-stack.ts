/**
 * DOVA Main CDK Stack
 */

import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { AuthConstruct } from './constructs/auth';
import { ApiConstruct } from './constructs/api';
import { StorageConstruct } from './constructs/storage';
import { MonitoringConstruct } from './constructs/monitoring';
import { DovaConfig } from './config';

export interface DovaStackProps extends cdk.StackProps, DovaConfig {}

export class DovaStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: DovaStackProps) {
    super(scope, id, props);

    // Authentication (Cognito)
    const auth = new AuthConstruct(this, 'Auth', {
      environment: props.environment,
      callbackUrls: props.cognitoCallbackUrls,
      logoutUrls: props.cognitoLogoutUrls,
    });

    // Storage (DynamoDB, S3, ElastiCache)
    const storage = new StorageConstruct(this, 'Storage', {
      environment: props.environment,
      redisNodeType: props.redisNodeType,
    });

    // API (API Gateway, Lambda)
    const api = new ApiConstruct(this, 'Api', {
      environment: props.environment,
      userPool: auth.userPool,
      userPoolClient: auth.userPoolClient,
      profileTable: storage.profileTable,
      cacheBucket: storage.cacheBucket,
      enableWaf: props.enableWaf,
      bedrockModelId: props.bedrockModelId,
    });

    // Monitoring (CloudWatch, Alarms)
    if (props.enableMonitoring) {
      new MonitoringConstruct(this, 'Monitoring', {
        environment: props.environment,
        apiGateway: api.api,
        profileTable: storage.profileTable,
      });
    }

    // Outputs
    new cdk.CfnOutput(this, 'ApiUrl', {
      value: api.api.url,
      description: 'DOVA API URL',
      exportName: `${props.environment}-dova-api-url`,
    });

    new cdk.CfnOutput(this, 'UserPoolId', {
      value: auth.userPool.userPoolId,
      description: 'Cognito User Pool ID',
      exportName: `${props.environment}-dova-user-pool-id`,
    });

    new cdk.CfnOutput(this, 'UserPoolClientId', {
      value: auth.userPoolClient.userPoolClientId,
      description: 'Cognito User Pool Client ID',
      exportName: `${props.environment}-dova-user-pool-client-id`,
    });

    new cdk.CfnOutput(this, 'CacheBucketName', {
      value: storage.cacheBucket.bucketName,
      description: 'S3 Cache Bucket Name',
      exportName: `${props.environment}-dova-cache-bucket`,
    });
  }
}
