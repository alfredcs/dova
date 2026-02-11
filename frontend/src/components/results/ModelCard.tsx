import { Download, Heart, ExternalLink } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'

interface ModelCardProps {
  model: Record<string, unknown>
}

export default function ModelCard({ model }: ModelCardProps) {
  const modelId = (model.modelId as string) || (model.id as string) || (model.name as string) || 'Unknown'
  const author = (model.author as string) || ''
  const pipelineTag = (model.pipeline_tag as string) || ''
  const libraryName = (model.library_name as string) || ''
  const tags: string[] = Array.isArray(model.tags) ? model.tags.map(String) : []
  const downloads = Number(model.downloads ?? 0)
  const likes = Number(model.likes ?? 0)
  const url = (model.url as string) || `https://huggingface.co/${modelId}`

  return (
    <Card className="transition-shadow hover:shadow-md">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="line-clamp-2 text-base font-medium">
            {modelId}
          </CardTitle>
          <Badge variant="huggingface">HF</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {author && (
          <div className="text-sm text-muted-foreground">
            <span className="font-medium">by {author}</span>
          </div>
        )}

        <div className="flex flex-wrap gap-1">
          {pipelineTag && (
            <Badge variant="secondary" className="text-xs">
              {pipelineTag}
            </Badge>
          )}
          {libraryName && (
            <Badge variant="outline" className="text-xs">
              {libraryName}
            </Badge>
          )}
          {tags.slice(0, 2).map((tag) => (
            <Badge key={tag} variant="outline" className="text-xs">
              {tag}
            </Badge>
          ))}
        </div>

        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <Download className="h-3 w-3" />
            {downloads.toLocaleString()}
          </span>
          <span className="flex items-center gap-1">
            <Heart className="h-3 w-3" />
            {likes.toLocaleString()}
          </span>
        </div>

        <div className="flex gap-2 pt-2">
          <Button variant="outline" size="sm" asChild>
            <a href={url} target="_blank" rel="noopener noreferrer">
              <ExternalLink className="mr-1 h-3 w-3" />
              View Model
            </a>
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
