import apiClient from './client';
import type { ResearchQuery, ResearchResponse, SearchResponse, SearchHistoryItem } from './types';

export async function conductResearch(query: ResearchQuery): Promise<ResearchResponse> {
  if (query.files && query.files.length > 0) {
    return conductResearchWithFiles(query);
  }
  const response = await apiClient.post<ResearchResponse>('/v1/research', query);
  return response.data;
}

async function conductResearchWithFiles(query: ResearchQuery): Promise<ResearchResponse> {
  const formData = new FormData();
  formData.append('query', query.query);
  if (query.sources) {
    formData.append('sources', JSON.stringify(query.sources));
  }
  if (query.max_results !== undefined) {
    formData.append('max_results', String(query.max_results));
  }
  if (query.orchestrator) {
    formData.append('orchestrator', query.orchestrator);
  }
  for (const file of query.files!) {
    formData.append('files', file);
  }
  const response = await apiClient.post<ResearchResponse>('/v1/research/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

export async function searchArxiv(query: string, maxResults = 10): Promise<SearchResponse> {
  const response = await apiClient.post<SearchResponse>('/v1/search/arxiv', {
    query,
    max_results: maxResults,
  });
  return response.data;
}

export async function searchGitHub(query: string, maxResults = 10): Promise<SearchResponse> {
  const response = await apiClient.post<SearchResponse>('/v1/search/github', {
    query,
    max_results: maxResults,
  });
  return response.data;
}

export async function searchHuggingFace(query: string, maxResults = 10): Promise<SearchResponse> {
  const response = await apiClient.post<SearchResponse>('/v1/search/huggingface', {
    query,
    max_results: maxResults,
  });
  return response.data;
}

export async function getSearchHistory(): Promise<SearchHistoryItem[]> {
  const response = await apiClient.get<SearchHistoryItem[]>('/v1/search/history');
  return response.data;
}
