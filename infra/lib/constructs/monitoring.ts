/**
 * Monitoring Construct - CloudWatch Dashboards and Alarms
 */

import * as cdk from 'aws-cdk-lib';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as sns from 'aws-cdk-lib/aws-sns';
import { Construct } from 'constructs';

export interface MonitoringConstructProps {
  environment: string;
  apiGateway: apigateway.RestApi;
  profileTable: dynamodb.Table;
}

export class MonitoringConstruct extends Construct {
  public readonly dashboard: cloudwatch.Dashboard;
  public readonly alarmTopic: sns.Topic;

  constructor(scope: Construct, id: string, props: MonitoringConstructProps) {
    super(scope, id);

    // SNS Topic for alarms
    this.alarmTopic = new sns.Topic(this, 'AlarmTopic', {
      topicName: `dova-${props.environment}-alarms`,
      displayName: 'DOVA Platform Alarms',
    });

    // CloudWatch Dashboard
    this.dashboard = new cloudwatch.Dashboard(this, 'DovaDashboard', {
      dashboardName: `dova-${props.environment}-dashboard`,
    });

    // API Gateway Metrics
    const apiLatencyMetric = new cloudwatch.Metric({
      namespace: 'AWS/ApiGateway',
      metricName: 'Latency',
      dimensionsMap: {
        ApiName: props.apiGateway.restApiName,
        Stage: props.environment,
      },
      statistic: 'p99',
      period: cdk.Duration.minutes(1),
    });

    const api4xxMetric = new cloudwatch.Metric({
      namespace: 'AWS/ApiGateway',
      metricName: '4XXError',
      dimensionsMap: {
        ApiName: props.apiGateway.restApiName,
        Stage: props.environment,
      },
      statistic: 'Sum',
      period: cdk.Duration.minutes(1),
    });

    const api5xxMetric = new cloudwatch.Metric({
      namespace: 'AWS/ApiGateway',
      metricName: '5XXError',
      dimensionsMap: {
        ApiName: props.apiGateway.restApiName,
        Stage: props.environment,
      },
      statistic: 'Sum',
      period: cdk.Duration.minutes(1),
    });

    const apiCountMetric = new cloudwatch.Metric({
      namespace: 'AWS/ApiGateway',
      metricName: 'Count',
      dimensionsMap: {
        ApiName: props.apiGateway.restApiName,
        Stage: props.environment,
      },
      statistic: 'Sum',
      period: cdk.Duration.minutes(1),
    });

    // DynamoDB Metrics
    const dynamoReadCapacity = new cloudwatch.Metric({
      namespace: 'AWS/DynamoDB',
      metricName: 'ConsumedReadCapacityUnits',
      dimensionsMap: {
        TableName: props.profileTable.tableName,
      },
      statistic: 'Sum',
      period: cdk.Duration.minutes(1),
    });

    const dynamoWriteCapacity = new cloudwatch.Metric({
      namespace: 'AWS/DynamoDB',
      metricName: 'ConsumedWriteCapacityUnits',
      dimensionsMap: {
        TableName: props.profileTable.tableName,
      },
      statistic: 'Sum',
      period: cdk.Duration.minutes(1),
    });

    // Add widgets to dashboard
    this.dashboard.addWidgets(
      new cloudwatch.TextWidget({
        markdown: `# DOVA Platform - ${props.environment.toUpperCase()}`,
        width: 24,
        height: 1,
      })
    );

    this.dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: 'API Requests',
        left: [apiCountMetric],
        width: 8,
        height: 6,
      }),
      new cloudwatch.GraphWidget({
        title: 'API Latency (p99)',
        left: [apiLatencyMetric],
        width: 8,
        height: 6,
      }),
      new cloudwatch.GraphWidget({
        title: 'API Errors',
        left: [api4xxMetric, api5xxMetric],
        width: 8,
        height: 6,
      })
    );

    this.dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: 'DynamoDB Read Capacity',
        left: [dynamoReadCapacity],
        width: 12,
        height: 6,
      }),
      new cloudwatch.GraphWidget({
        title: 'DynamoDB Write Capacity',
        left: [dynamoWriteCapacity],
        width: 12,
        height: 6,
      })
    );

    // Alarms
    const highLatencyAlarm = new cloudwatch.Alarm(this, 'HighLatencyAlarm', {
      alarmName: `dova-${props.environment}-high-latency`,
      alarmDescription: 'API latency is above threshold',
      metric: apiLatencyMetric,
      threshold: 5000, // 5 seconds
      evaluationPeriods: 3,
      comparisonOperator:
        cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
    });
    highLatencyAlarm.addAlarmAction(
      new cdk.aws_cloudwatch_actions.SnsAction(this.alarmTopic)
    );

    const high5xxAlarm = new cloudwatch.Alarm(this, 'High5xxAlarm', {
      alarmName: `dova-${props.environment}-high-5xx`,
      alarmDescription: 'High number of 5xx errors',
      metric: api5xxMetric,
      threshold: 10,
      evaluationPeriods: 2,
      comparisonOperator:
        cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
    });
    high5xxAlarm.addAlarmAction(
      new cdk.aws_cloudwatch_actions.SnsAction(this.alarmTopic)
    );
  }
}
