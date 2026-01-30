import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Skeleton } from '@/components/ui/skeleton'
import SynthesisSummary from './SynthesisSummary'
import PaperCard from './PaperCard'
import RepoCard from './RepoCard'
import ModelCard from './ModelCard'
import type { ResearchResponse, ArxivPaper, GitHubRepo, HuggingFaceModel } from '@/api/types'

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

  // Extract results by type
  const papers: ArxivPaper[] = []
  const repos: GitHubRepo[] = []
  const models: HuggingFaceModel[] = []

  data.results.forEach((result) => {
    if (result.papers) papers.push(...result.papers)
    if (result.repositories) repos.push(...result.repositories)
    if (result.models) models.push(...result.models)
  })

  const totalCount = papers.length + repos.length + models.length

  return (
    <div className="space-y-6">
      {data.synthesis && <SynthesisSummary synthesis={data.synthesis} />}

      <Tabs defaultValue="all">
        <TabsList>
          <TabsTrigger value="all">All ({totalCount})</TabsTrigger>
          <TabsTrigger value="papers">Papers ({papers.length})</TabsTrigger>
          <TabsTrigger value="repos">Repos ({repos.length})</TabsTrigger>
          <TabsTrigger value="models">Models ({models.length})</TabsTrigger>
        </TabsList>

        <TabsContent value="all" className="mt-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {papers.map((paper) => (
              <PaperCard key={paper.id} paper={paper} />
            ))}
            {repos.map((repo) => (
              <RepoCard key={repo.id} repo={repo} />
            ))}
            {models.map((model) => (
              <ModelCard key={model.id} model={model} />
            ))}
          </div>
        </TabsContent>

        <TabsContent value="papers" className="mt-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {papers.map((paper) => (
              <PaperCard key={paper.id} paper={paper} />
            ))}
          </div>
          {papers.length === 0 && (
            <p className="py-8 text-center text-muted-foreground">
              No papers found
            </p>
          )}
        </TabsContent>

        <TabsContent value="repos" className="mt-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {repos.map((repo) => (
              <RepoCard key={repo.id} repo={repo} />
            ))}
          </div>
          {repos.length === 0 && (
            <p className="py-8 text-center text-muted-foreground">
              No repositories found
            </p>
          )}
        </TabsContent>

        <TabsContent value="models" className="mt-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {models.map((model) => (
              <ModelCard key={model.id} model={model} />
            ))}
          </div>
          {models.length === 0 && (
            <p className="py-8 text-center text-muted-foreground">
              No models found
            </p>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}
