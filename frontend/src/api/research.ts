import apiClient from './client';
import type { ResearchQuery, ResearchResponse, SearchResponse, SearchHistoryItem } from './types';

export async function conductResearch(query: ResearchQuery): Promise<ResearchResponse> {
  const response = await apiClient.post<ResearchResponse>('/v1/research', query);
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
