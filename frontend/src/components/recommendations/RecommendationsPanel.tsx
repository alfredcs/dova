import { Lightbulb, TrendingUp } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import type { Recommendation } from '@/api/types'

interface RecommendationsPanelProps {
  recommendations: Recommendation[]
  onSelect?: (recommendation: Recommendation) => void
}

export default function RecommendationsPanel({
  recommendations,
  onSelect,
}: RecommendationsPanelProps) {
  if (recommendations.length === 0) {
    return null
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Lightbulb className="h-4 w-4 text-yellow-500" />
          Recommended for You
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {recommendations.map((rec) => (
          <button
            key={rec.id}
            onClick={() => onSelect?.(rec)}
            className="w-full rounded-md border p-3 text-left transition-colors hover:bg-accent"
          >
            <div className="flex items-start justify-between gap-2">
              <span className="text-sm font-medium">{rec.topic}</span>
              <Badge
                variant={
                  rec.source_type === 'arxiv'
                    ? 'arxiv'
                    : rec.source_type === 'github'
                    ? 'github'
                    : 'huggingface'
                }
                className="text-xs"
              >
                {rec.source_type}
              </Badge>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">{rec.reason}</p>
            <div className="mt-2 flex items-center gap-1 text-xs text-muted-foreground">
              <TrendingUp className="h-3 w-3" />
              <span>{Math.round(rec.relevance_score * 100)}% match</span>
            </div>
          </button>
        ))}
      </CardContent>
    </Card>
  )
}
