/**
 * Storage Construct - DynamoDB, S3, ElastiCache
 */

import * as cdk from 'aws-cdk-lib';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as elasticache from 'aws-cdk-lib/aws-elasticache';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { Construct } from 'constructs';

export interface StorageConstructProps {
  environment: string;
  redisNodeType?: string;
}

export class StorageConstruct extends Construct {
  public readonly profileTable: dynamodb.Table;
  public readonly researchCacheTable: dynamodb.Table;
  public readonly cacheBucket: s3.Bucket;
  public readonly redisCluster?: elasticache.CfnCacheCluster;

  constructor(scope: Construct, id: string, props: StorageConstructProps) {
    super(scope, id);

    const isProd = props.environment === 'production';

    // User Profile Table
    this.profileTable = new dynamodb.Table(this, 'ProfileTable', {
      tableName: `dova-${props.environment}-profiles`,
      partitionKey: {
        name: 'user_id',
        type: dynamodb.AttributeType.STRING,
      },
      billingMode: isProd
        ? dynamodb.BillingMode.PROVISIONED
        : dynamodb.BillingMode.PAY_PER_REQUEST,
      readCapacity: isProd ? 10 : undefined,
      writeCapacity: isProd ? 10 : undefined,
      pointInTimeRecovery: isProd,
      removalPolicy: isProd
        ? cdk.RemovalPolicy.RETAIN
        : cdk.RemovalPolicy.DESTROY,
      timeToLiveAttribute: 'ttl',
    });

    // Add GSI for querying by email
    this.profileTable.addGlobalSecondaryIndex({
      indexName: 'email-index',
      partitionKey: {
        name: 'email',
        type: dynamodb.AttributeType.STRING,
      },
      projectionType: dynamodb.ProjectionType.ALL,
    });

    // Research Cache Table
    this.researchCacheTable = new dynamodb.Table(this, 'ResearchCacheTable', {
      tableName: `dova-${props.environment}-research-cache`,
      partitionKey: {
        name: 'cache_key',
        type: dynamodb.AttributeType.STRING,
      },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      timeToLiveAttribute: 'expires_at',
    });

    // S3 Bucket for artifacts and cache
    this.cacheBucket = new s3.Bucket(this, 'CacheBucket', {
      bucketName: `dova-${props.environment}-cache-${cdk.Aws.ACCOUNT_ID}`,
      versioned: false,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: isProd
        ? cdk.RemovalPolicy.RETAIN
        : cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: !isProd,
      lifecycleRules: [
        {
          id: 'ExpireOldCache',
          expiration: cdk.Duration.days(30),
          prefix: 'cache/',
        },
        {
          id: 'ExpireOldArtifacts',
          expiration: cdk.Duration.days(90),
          prefix: 'artifacts/',
        },
      ],
      cors: [
        {
          allowedMethods: [s3.HttpMethods.GET, s3.HttpMethods.PUT],
          allowedOrigins: ['*'],
          allowedHeaders: ['*'],
          maxAge: 3000,
        },
      ],
    });

    // ElastiCache Redis (only for staging/production)
    if (props.environment !== 'development' && props.redisNodeType) {
      // Note: In a real deployment, you'd need a VPC
      // This is a simplified example
      const vpc = new ec2.Vpc(this, 'DovaVpc', {
        maxAzs: 2,
        natGateways: 1,
      });

      const redisSecurityGroup = new ec2.SecurityGroup(
        this,
        'RedisSecurityGroup',
        {
          vpc,
          description: 'Security group for DOVA Redis cluster',
          allowAllOutbound: true,
        }
      );

      redisSecurityGroup.addIngressRule(
        ec2.Peer.ipv4(vpc.vpcCidrBlock),
        ec2.Port.tcp(6379),
        'Allow Redis access from VPC'
      );

      const subnetGroup = new elasticache.CfnSubnetGroup(
        this,
        'RedisSubnetGroup',
        {
          description: 'Subnet group for DOVA Redis',
          subnetIds: vpc.privateSubnets.map((s) => s.subnetId),
          cacheSubnetGroupName: `dova-${props.environment}-redis-subnet`,
        }
      );

      this.redisCluster = new elasticache.CfnCacheCluster(
        this,
        'RedisCluster',
        {
          clusterName: `dova-${props.environment}-redis`,
          engine: 'redis',
          cacheNodeType: props.redisNodeType,
          numCacheNodes: 1,
          vpcSecurityGroupIds: [redisSecurityGroup.securityGroupId],
          cacheSubnetGroupName: subnetGroup.cacheSubnetGroupName,
          engineVersion: '7.0',
          port: 6379,
          autoMinorVersionUpgrade: true,
        }
      );

      this.redisCluster.addDependency(subnetGroup);
    }
  }
}
