import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Skeleton } from '@/components/ui/skeleton'
import SynthesisSummary from './SynthesisSummary'
import PaperCard from './PaperCard'
import RepoCard from './RepoCard'
import ModelCard from './ModelCard'
import WebResultCard from './WebResultCard'
import ImageGallery from './ImageGallery'
import type { ResearchResponse, ImageResult } from '@/api/types'

interface ResultsPanelProps {
  data?: ResearchResponse
  isLoading?: boolean
}

function ResultsSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-40 w-full" />
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {[...Array(6)].map((_, i) => (
          <Skeleton key={i} className="h-48" />
        ))}
      </div>
    </div>
  )
}

export default function ResultsPanel({ data, isLoading }: ResultsPanelProps) {
  if (isLoading) {
    return <ResultsSkeleton />
  }

  if (!data) {
    return (
      <div className="flex h-64 items-center justify-center text-muted-foreground">
        Enter a research query to get started
      </div>
    )
  }

  // Backend returns flat arrays directly on the response
  const papers = data.papers || []
  const repos = data.repositories || []
  const models = data.models || []
  const datasets = data.datasets || []
  const webResults = data.web_results || []
  const images: ImageResult[] = data.images || []

  const totalCount = papers.length + repos.length + models.length + datasets.length + webResults.length

  // Build visible tabs — only show tabs that have results
  const tabs: { value: string; label: string; count: number }[] = []
  if (papers.length > 0) tabs.push({ value: 'papers', label: 'Papers', count: papers.length })
  if (repos.length > 0) tabs.push({ value: 'repos', label: 'Repos', count: repos.length })
  if (models.length > 0) tabs.push({ value: 'models', label: 'Models', count: models.length })
  if (datasets.length > 0) tabs.push({ value: 'datasets', label: 'Datasets', count: datasets.length })
  if (webResults.length > 0) tabs.push({ value: 'web', label: 'Web', count: webResults.length })

  return (
    <div className="space-y-6">
      {images.length > 0 && <ImageGallery images={images} />}

      {(data.answer || data.summary) && (
        <SynthesisSummary
          answer={data.answer}
          summary={data.summary}
          insights={data.insights}
          recommendations={data.recommendations}
          confidence={data.confidence}
        />
      )}

      {totalCount > 0 && (
        <Tabs defaultValue="all">
          <TabsList>
            <TabsTrigger value="all">All ({totalCount})</TabsTrigger>
            {tabs.map((tab) => (
              <TabsTrigger key={tab.value} value={tab.value}>
                {tab.label} ({tab.count})
              </TabsTrigger>
            ))}
          </TabsList>

          <TabsContent value="all" className="mt-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {papers.map((paper, i) => (
                <PaperCard key={`paper-${(paper as Record<string, unknown>).id || (paper as Record<string, unknown>).arxiv_id || i}`} paper={paper as any} />
              ))}
              {repos.map((repo, i) => (
                <RepoCard key={`repo-${(repo as Record<string, unknown>).id || (repo as Record<string, unknown>).full_name || (repo as Record<string, unknown>).name || i}`} repo={repo as any} />
              ))}
              {models.map((model, i) => (
                <ModelCard key={`model-${(model as Record<string, unknown>).id || (model as Record<string, unknown>).modelId || i}`} model={model as any} />
              ))}
              {datasets.map((ds, i) => (
                <ModelCard key={`dataset-${(ds as Record<string, unknown>).id || i}`} model={ds as any} />
              ))}
              {webResults.map((wr, i) => (
                <WebResultCard key={`web-${(wr as Record<string, unknown>).url || i}`} result={wr as any} />
              ))}
            </div>
          </TabsContent>

          <TabsContent value="papers" className="mt-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {papers.map((paper, i) => (
                <PaperCard key={`paper-${(paper as Record<string, unknown>).id || (paper as Record<string, unknown>).arxiv_id || i}`} paper={paper as any} />
              ))}
            </div>
          </TabsContent>

          <TabsContent value="repos" className="mt-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {repos.map((repo, i) => (
                <RepoCard key={`repo-${(repo as Record<string, unknown>).id || (repo as Record<string, unknown>).full_name || (repo as Record<string, unknown>).name || i}`} repo={repo as any} />
              ))}
            </div>
          </TabsContent>

          <TabsContent value="models" className="mt-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {models.map((model, i) => (
                <ModelCard key={`model-${(model as Record<string, unknown>).id || (model as Record<string, unknown>).modelId || i}`} model={model as any} />
              ))}
            </div>
          </TabsContent>

          <TabsContent value="datasets" className="mt-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {datasets.map((ds, i) => (
                <ModelCard key={`dataset-${(ds as Record<string, unknown>).id || i}`} model={ds as any} />
              ))}
            </div>
          </TabsContent>

          <TabsContent value="web" className="mt-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {webResults.map((wr, i) => (
                <WebResultCard key={`web-${(wr as Record<string, unknown>).url || i}`} result={wr as any} />
              ))}
            </div>
          </TabsContent>
        </Tabs>
      )}

      {totalCount === 0 && !data.answer && !data.summary && (
        <div className="flex h-32 items-center justify-center text-muted-foreground">
          No results found. Try adjusting your query or enabling more sources.
        </div>
      )}
    </div>
  )
}
