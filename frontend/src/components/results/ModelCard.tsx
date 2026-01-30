import { Download, Heart, ExternalLink } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { HuggingFaceModel } from '@/api/types'

interface ModelCardProps {
  model: HuggingFaceModel
}

export default function ModelCard({ model }: ModelCardProps) {
  const modelUrl = `https://huggingface.co/${model.modelId}`

  return (
    <Card className="transition-shadow hover:shadow-md">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="line-clamp-2 text-base font-medium">
            {model.modelId}
          </CardTitle>
          <Badge variant="huggingface">HF</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="text-sm text-muted-foreground">
          <span className="font-medium">by {model.author}</span>
        </div>

        <div className="flex flex-wrap gap-1">
          {model.pipeline_tag && (
            <Badge variant="secondary" className="text-xs">
              {model.pipeline_tag}
            </Badge>
          )}
          {model.library_name && (
            <Badge variant="outline" className="text-xs">
              {model.library_name}
            </Badge>
          )}
          {model.tags.slice(0, 2).map((tag) => (
            <Badge key={tag} variant="outline" className="text-xs">
              {tag}
            </Badge>
          ))}
        </div>

        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <Download className="h-3 w-3" />
            {model.downloads.toLocaleString()}
          </span>
          <span className="flex items-center gap-1">
            <Heart className="h-3 w-3" />
            {model.likes.toLocaleString()}
          </span>
        </div>

        <div className="flex gap-2 pt-2">
          <Button variant="outline" size="sm" asChild>
            <a href={modelUrl} target="_blank" rel="noopener noreferrer">
              <ExternalLink className="mr-1 h-3 w-3" />
              View Model
            </a>
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
