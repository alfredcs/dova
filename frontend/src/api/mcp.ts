import apiClient from './client';
import type { MCPServerListResponse } from './types';

export async function getMCPServers(checkHealth = false): Promise<MCPServerListResponse> {
  const { data } = await apiClient.get('/v1/mcp/servers', {
    params: { check_health: checkHealth },
  });
  return data;
}
