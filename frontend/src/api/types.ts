// Research types
export interface ResearchQuery {
  query: string;
  sources?: string[];
  max_results?: number;
  orchestrator?: 'standard' | 'thinking';
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

export interface ImageResult {
  url: string;
  prompt: string;
  resolution: string;
  seed: number;
}

export interface ResearchResponse {
  query: string;
  status: string;
  answer: string;
  summary: string;
  papers: Record<string, unknown>[];
  repositories: Record<string, unknown>[];
  models: Record<string, unknown>[];
  datasets: Record<string, unknown>[];
  web_results: Record<string, unknown>[];
  images: ImageResult[];
  insights: string[];
  recommendations: string[];
  confidence: number;
  refinement_attempts: number;
  reasoning_trace: Record<string, unknown>[];
  debate: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

// Search types (source-specific search endpoints)
export interface SearchResponse {
  source: string;
  query: string;
  results: Record<string, unknown>[];
  total_count: number;
  metadata: Record<string, unknown>;
}

// Chat types
export interface ThinkingStep {
  step_type: string;
  content: string;
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  sources?: string[];
  show_thinking?: boolean;
  orchestrator?: 'standard' | 'thinking';
}

export interface ChatResponse {
  session_id: string;
  message: string;
  thinking: ThinkingStep[];
  action_taken?: string;
  sources_used: string[];
  research_results?: Record<string, unknown>;
  debate_results?: Record<string, unknown>;
  images: ImageResult[];
  metadata: Record<string, unknown>;
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

// MCP Server types
export interface MCPServer {
  name: string;
  description?: string;
  transport: 'stdio' | 'http' | 'sse';
  enabled: boolean;
  url?: string;
  command?: string;
  status: 'healthy' | 'unhealthy' | 'unknown';
  status_message?: string;
}

export interface MCPServerListResponse {
  servers: MCPServer[];
  total: number;
  timestamp: string;
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
