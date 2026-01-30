import apiClient from './client';
import {
  MemorySearchResponse,
  KnowledgeItem,
  PromoteToKnowledgeRequest,
} from './types';

export async function searchMemory(
  query: string,
  type = 'all',
  maxResults = 10
): Promise<MemorySearchResponse> {
  const { data } = await apiClient.get('/v1/memory/search', {
    params: { q: query, type, max_results: maxResults },
  });
  return data;
}

export async function getHistory(): Promise<MemorySearchResponse> {
  const { data } = await apiClient.get('/v1/memory/history');
  return data;
}

export async function getKnowledge(): Promise<KnowledgeItem[]> {
  const { data } = await apiClient.get('/v1/memory/knowledge');
  return data;
}

export async function promoteToKnowledge(
  req: PromoteToKnowledgeRequest
): Promise<KnowledgeItem> {
  const { data } = await apiClient.post('/v1/memory/promote', req);
  return data;
}

export async function deleteMemory(memoryId: string): Promise<void> {
  await apiClient.delete(`/v1/memory/${memoryId}`);
}
