import { Search, Clock, Filter } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { useSearchHistory } from '@/hooks/useResearch'

export default function History() {
  const { data: history, isLoading, error } = useSearchHistory()

  if (isLoading) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-bold">Search History</h1>
        {[...Array(5)].map((_, i) => (
          <Skeleton key={i} className="h-24" />
        ))}
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex h-64 items-center justify-center text-muted-foreground">
        Failed to load search history. Please try again.
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Search History</h1>

      {history && history.length > 0 ? (
        <div className="space-y-4">
          {history.map((item) => (
            <Card key={item.id} className="transition-shadow hover:shadow-md">
              <CardContent className="flex items-center justify-between py-4">
                <div className="flex items-center gap-4">
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10">
                    <Search className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <h3 className="font-medium">{item.query}</h3>
                    <div className="mt-1 flex items-center gap-4 text-xs text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {new Date(item.timestamp).toLocaleString()}
                      </span>
                      <span className="flex items-center gap-1">
                        <Filter className="h-3 w-3" />
                        {item.results_count} results
                      </span>
                    </div>
                  </div>
                </div>
                <div className="flex gap-1">
                  {item.sources.map((source) => (
                    <Badge
                      key={source}
                      variant={
                        source === 'arxiv'
                          ? 'arxiv'
                          : source === 'github'
                          ? 'github'
                          : 'huggingface'
                      }
                      className="text-xs"
                    >
                      {source}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <Card>
          <CardContent className="flex h-64 flex-col items-center justify-center text-muted-foreground">
            <Search className="mb-4 h-12 w-12" />
            <p>No search history yet</p>
            <p className="text-sm">Your searches will appear here</p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
