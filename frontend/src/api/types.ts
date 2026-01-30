// Research types
export interface ResearchQuery {
  query: string;
  sources?: string[];
  max_results?: number;
}

export interface ArxivPaper {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  published: string;
  updated: string;
  categories: string[];
  pdf_url: string;
  arxiv_url: string;
}

export interface GitHubRepo {
  id: number;
  name: string;
  full_name: string;
  description: string;
  html_url: string;
  stargazers_count: number;
  forks_count: number;
  language: string;
  topics: string[];
  updated_at: string;
}

export interface HuggingFaceModel {
  id: string;
  modelId: string;
  author: string;
  tags: string[];
  downloads: number;
  likes: number;
  pipeline_tag?: string;
  library_name?: string;
}

export interface ResearchResult {
  source: 'arxiv' | 'github' | 'huggingface';
  papers?: ArxivPaper[];
  repositories?: GitHubRepo[];
  models?: HuggingFaceModel[];
}

export interface SynthesisSummary {
  summary: string;
  key_findings: string[];
  recommended_actions: string[];
}

export interface ResearchResponse {
  query: string;
  synthesis: SynthesisSummary;
  results: ResearchResult[];
  timestamp: string;
}

// Profile types
export interface UserInterest {
  topic: string;
  weight: number;
  added_at: string;
}

export interface UserProfile {
  id: string;
  email: string;
  name: string;
  interests: UserInterest[];
  expertise_level: 'beginner' | 'intermediate' | 'expert';
  preferred_sources: string[];
  created_at: string;
  updated_at: string;
}

export interface ProfileUpdateRequest {
  name?: string;
  interests?: UserInterest[];
  expertise_level?: 'beginner' | 'intermediate' | 'expert';
  preferred_sources?: string[];
}

// Recommendations types
export interface Recommendation {
  id: string;
  topic: string;
  reason: string;
  relevance_score: number;
  source_type: 'arxiv' | 'github' | 'huggingface';
  preview?: ArxivPaper | GitHubRepo | HuggingFaceModel;
}

export interface RecommendationsResponse {
  recommendations: Recommendation[];
  generated_at: string;
}

// Search history types
export interface SearchHistoryItem {
  id: string;
  query: string;
  sources: string[];
  results_count: number;
  timestamp: string;
}

// Memory types
export interface MemoryEntry {
  id: string;
  type: 'short_term' | 'long_term' | 'knowledge';
  content: Record<string, unknown>;
  created_at: string;
  expires_at?: string;
  summary_text: string;
}

export interface KnowledgeItem {
  id: string;
  topic: string;
  summary: string;
  source_sessions: string[];
  promoted_at: string;
}

export interface MemorySearchResponse {
  entries: MemoryEntry[];
  total_count: number;
}

export interface PromoteToKnowledgeRequest {
  topic: string;
  summary: string;
  session_ids: string[];
}

// Source types
export interface QualityMetrics {
  query_count: number;
  click_count: number;
  save_count: number;
  quality_score: number;
}

export interface Source {
  id: string;
  name: string;
  source_type: 'builtin' | 'web_url' | 'rss_feed' | 'api';
  enabled: boolean;
  quality: QualityMetrics;
  created_at: string;
}

export interface SourceConfig {
  url: string;
  headers?: Record<string, string>;
  auth_type?: 'bearer' | 'api_key' | 'basic';
  auth_value?: string;
  content_selector?: string;
}

export interface CreateSourceRequest {
  name: string;
  source_type: 'web_url' | 'rss_feed' | 'api';
  config: SourceConfig;
}
