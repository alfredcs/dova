import { useState, useMemo } from 'react'
import SearchBar from '@/components/search/SearchBar'
import SearchFilters, { expandSourceGroups } from '@/components/search/SearchFilters'
import ResultsPanel from '@/components/results/ResultsPanel'
import ProgressTimeline from '@/components/results/ProgressTimeline'
import RecommendationsPanel from '@/components/recommendations/RecommendationsPanel'
import { useResearchStream } from '@/hooks/useResearchStream'
import { useRecommendations } from '@/hooks/useRecommendations'
import type { Recommendation, ResearchResponse } from '@/api/types'

export default function Dashboard() {
  // Group IDs (ai / web / bio). All three are enabled by default so the
  // orchestrator considers every pool during deliberation. Groups expand
  // to concrete source names (arxiv, github, huggingface, ...) at the
  // API boundary via `expandSourceGroups`.
  const [selectedSources, setSelectedSources] = useState<string[]>([
    'ai',
    'web',
    'bio',
  ])
  const [orchestrator, setOrchestrator] = useState<'standard' | 'thinking'>('thinking')

  const {
    search,
    status,
    stageMessage,
    partialResults,
    transactionLog,
    finalResult,
    streamingAnswer,
    error,
    isStreaming,
    hasResults,
  } = useResearchStream()
  const { data: recommendationsData } = useRecommendations()

  // Build a ResearchResponse-shaped object from partial results during streaming
  const displayData: ResearchResponse | undefined = useMemo(() => {
    if (finalResult) return finalResult
    if (!isStreaming) return undefined
    return {
      query: '',
      status: 'streaming',
      answer: streamingAnswer,
      summary: '',
      papers: partialResults.papers,
      repositories: partialResults.repositories,
      models: partialResults.models,
      datasets: partialResults.datasets,
      web_results: partialResults.web_results,
      images: partialResults.images as any,
      insights: [],
      recommendations: [],
      confidence: 0,
      refinement_attempts: 0,
      reasoning_trace: [],
      debate: {},
      metadata: {},
    }
  }, [finalResult, isStreaming, partialResults, streamingAnswer])

  const handleSearch = (query: string, files: File[] = []) => {
    search({
      query,
      sources: expandSourceGroups(selectedSources),
      max_results: 10,
      orchestrator,
      files: files.length > 0 ? files : undefined,
    })
  }

  const handleSourceToggle = (source: string) => {
    setSelectedSources((prev) =>
      prev.includes(source)
        ? prev.filter((s) => s !== source)
        : [...prev, source]
    )
  }

  const handleRecommendationSelect = (rec: Recommendation) => {
    search({
      query: rec.topic,
      sources: [rec.source_type],
      max_results: 10,
      orchestrator,
    })
  }

  // Only show skeleton when streaming and no partial results have arrived yet
  const isLoading = isStreaming && !hasResults

  return (
    <div className="space-y-6">
      {/* Search Section */}
      <div className="space-y-4">
        <SearchBar onSearch={handleSearch} isLoading={isStreaming} />
        <SearchFilters
          selectedSources={selectedSources}
          onToggle={handleSourceToggle}
          orchestrator={orchestrator}
          onOrchestratorChange={setOrchestrator}
        />
      </div>

      {/* Error Display */}
      {status === 'error' && error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300">
          <p className="font-medium">Research failed</p>
          <p className="mt-1">{error}</p>
        </div>
      )}

      {/* Progress Timeline */}
      {(isStreaming || (status === 'complete' && transactionLog.length > 0)) && (
        <ProgressTimeline
          log={transactionLog}
          stageMessage={stageMessage}
          isStreaming={isStreaming}
        />
      )}

      {/* Main Content */}
      <div className="grid gap-6 lg:grid-cols-4">
        <div className="lg:col-span-3">
          <ResultsPanel data={displayData} isLoading={isLoading} />
        </div>
        <div className="space-y-4">
          {recommendationsData && (
            <RecommendationsPanel
              recommendations={recommendationsData.recommendations}
              onSelect={handleRecommendationSelect}
            />
          )}
        </div>
      </div>
    </div>
  )
}
