import { useState } from 'react'
import SearchBar from '@/components/search/SearchBar'
import SearchFilters from '@/components/search/SearchFilters'
import ResultsPanel from '@/components/results/ResultsPanel'
import RecommendationsPanel from '@/components/recommendations/RecommendationsPanel'
import { useResearch } from '@/hooks/useResearch'
import { useRecommendations } from '@/hooks/useRecommendations'
import type { Recommendation } from '@/api/types'

export default function Dashboard() {
  const [selectedSources, setSelectedSources] = useState([
    'arxiv',
    'github',
    'huggingface',
  ])

  const { mutate: search, data: results, isPending: isSearching } = useResearch()
  const { data: recommendationsData } = useRecommendations()

  const handleSearch = (query: string) => {
    search({
      query,
      sources: selectedSources,
      max_results: 10,
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
    })
  }

  return (
    <div className="space-y-6">
      {/* Search Section */}
      <div className="space-y-4">
        <SearchBar onSearch={handleSearch} isLoading={isSearching} />
        <SearchFilters
          selectedSources={selectedSources}
          onToggle={handleSourceToggle}
        />
      </div>

      {/* Main Content */}
      <div className="grid gap-6 lg:grid-cols-4">
        <div className="lg:col-span-3">
          <ResultsPanel data={results} isLoading={isSearching} />
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
