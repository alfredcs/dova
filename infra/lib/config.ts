/**
 * DOVA Infrastructure Configuration
 */

export interface DovaConfig {
  environment: string;
  apiDomainName?: string;
  enableWaf?: boolean;
  enableMonitoring?: boolean;
  redisNodeType?: string;
  cognitoCallbackUrls?: string[];
  cognitoLogoutUrls?: string[];
  bedrockModelId?: string;
}

export const config: Record<string, DovaConfig> = {
  development: {
    environment: 'development',
    enableWaf: false,
    enableMonitoring: false,
    redisNodeType: 'cache.t3.micro',
    cognitoCallbackUrls: ['http://localhost:3000/callback'],
    cognitoLogoutUrls: ['http://localhost:3000'],
    bedrockModelId: 'anthropic.claude-sonnet-4-20250514-v1:0',
  },
  staging: {
    environment: 'staging',
    enableWaf: true,
    enableMonitoring: true,
    redisNodeType: 'cache.t3.small',
    cognitoCallbackUrls: ['https://staging.dova.example.com/callback'],
    cognitoLogoutUrls: ['https://staging.dova.example.com'],
    bedrockModelId: 'anthropic.claude-sonnet-4-20250514-v1:0',
  },
  production: {
    environment: 'production',
    enableWaf: true,
    enableMonitoring: true,
    redisNodeType: 'cache.r6g.large',
    cognitoCallbackUrls: ['https://dova.example.com/callback'],
    cognitoLogoutUrls: ['https://dova.example.com'],
    bedrockModelId: 'anthropic.claude-sonnet-4-20250514-v1:0',
  },
};
