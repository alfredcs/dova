#!/usr/bin/env node
/**
 * DOVA CDK Application Entry Point
 */

import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { DovaStack } from '../lib/dova-stack';
import { config } from '../lib/config';

const app = new cdk.App();

// Get environment from context or default
const environment = app.node.tryGetContext('environment') || 'development';

new DovaStack(app, `DovaStack-${environment}`, {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || 'us-east-1',
  },
  stackName: `dova-${environment}`,
  description: 'DOVA - Deep Orchestrated Versatile Agent Platform',
  tags: {
    Project: 'DOVA',
    Environment: environment,
    ManagedBy: 'CDK',
  },
  ...config[environment as keyof typeof config],
});

app.synth();
