import apiClient from './client';
import type { Source, CreateSourceRequest } from './types';

export async function getSources(enabledOnly = false): Promise<Source[]> {
  const { data } = await apiClient.get('/v1/sources', {
    params: { enabled_only: enabledOnly },
  });
  return data;
}

export async function createSource(req: CreateSourceRequest): Promise<Source> {
  const { data } = await apiClient.post('/v1/sources', req);
  return data;
}

export async function updateSource(
  sourceId: string,
  updates: { name?: string; enabled?: boolean }
): Promise<Source> {
  const { data } = await apiClient.put(`/v1/sources/${sourceId}`, updates);
  return data;
}

export async function deleteSource(sourceId: string): Promise<void> {
  await apiClient.delete(`/v1/sources/${sourceId}`);
}

export async function recordInteraction(
  sourceId: string,
  type: 'query' | 'click' | 'save',
  position?: number
): Promise<void> {
  await apiClient.post('/v1/sources/interact', {
    source_id: sourceId,
    interaction_type: type,
    result_position: position,
  });
}
