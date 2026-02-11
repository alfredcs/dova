import { Globe, ExternalLink } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'

interface WebResultCardProps {
  result: Record<string, unknown>
}

export default function WebResultCard({ result }: WebResultCardProps) {
  const title = (result.title as string) || 'Untitled'
  const description = (result.description as string) || (result.snippet as string) || (result.content as string) || ''
  const url = (result.url as string) || (result.link as string) || ''
  const source = (result.source as string) || (url ? new URL(url).hostname : '')

  return (
    <Card className="transition-shadow hover:shadow-md">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="line-clamp-2 text-base font-medium">
            {title}
          </CardTitle>
          <Badge variant="outline" className="text-blue-600 border-blue-300">Web</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="line-clamp-3 text-sm text-muted-foreground">
          {description}
        </p>

        {source && (
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <Globe className="h-3 w-3" />
            {source}
          </div>
        )}

        {url && (
          <div className="flex gap-2 pt-2">
            <Button variant="outline" size="sm" asChild>
              <a href={url} target="_blank" rel="noopener noreferrer">
                <ExternalLink className="mr-1 h-3 w-3" />
                Visit
              </a>
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
