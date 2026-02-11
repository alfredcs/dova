import { Star, GitFork, Code, ExternalLink } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'

interface RepoCardProps {
  repo: Record<string, unknown>
}

export default function RepoCard({ repo }: RepoCardProps) {
  const fullName = (repo.full_name as string) || (repo.name as string) || 'Unknown'
  const description = (repo.description as string) || 'No description available'
  const htmlUrl = (repo.html_url as string) || (repo.url as string) || ''
  const stars = Number(repo.stargazers_count ?? repo.stars ?? 0)
  const forks = Number(repo.forks_count ?? repo.forks ?? 0)
  const language = (repo.language as string) || ''
  const topics: string[] = Array.isArray(repo.topics) ? repo.topics.map(String) : []

  return (
    <Card className="transition-shadow hover:shadow-md">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="line-clamp-2 text-base font-medium">
            {fullName}
          </CardTitle>
          <Badge variant="github">GitHub</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="line-clamp-3 text-sm text-muted-foreground">
          {description}
        </p>

        {topics.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {topics.slice(0, 4).map((topic) => (
              <Badge key={topic} variant="outline" className="text-xs">
                {topic}
              </Badge>
            ))}
          </div>
        )}

        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <Star className="h-3 w-3" />
            {stars.toLocaleString()}
          </span>
          {forks > 0 && (
            <span className="flex items-center gap-1">
              <GitFork className="h-3 w-3" />
              {forks.toLocaleString()}
            </span>
          )}
          {language && (
            <span className="flex items-center gap-1">
              <Code className="h-3 w-3" />
              {language}
            </span>
          )}
        </div>

        {htmlUrl && (
          <div className="flex gap-2 pt-2">
            <Button variant="outline" size="sm" asChild>
              <a href={htmlUrl} target="_blank" rel="noopener noreferrer">
                <ExternalLink className="mr-1 h-3 w-3" />
                View Repository
              </a>
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
