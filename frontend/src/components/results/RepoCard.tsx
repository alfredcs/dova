import { Star, GitFork, Code, ExternalLink } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { GitHubRepo } from '@/api/types'

interface RepoCardProps {
  repo: GitHubRepo
}

export default function RepoCard({ repo }: RepoCardProps) {
  return (
    <Card className="transition-shadow hover:shadow-md">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="line-clamp-2 text-base font-medium">
            {repo.full_name}
          </CardTitle>
          <Badge variant="github">GitHub</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="line-clamp-3 text-sm text-muted-foreground">
          {repo.description || 'No description available'}
        </p>

        <div className="flex flex-wrap gap-1">
          {repo.topics.slice(0, 4).map((topic) => (
            <Badge key={topic} variant="outline" className="text-xs">
              {topic}
            </Badge>
          ))}
        </div>

        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <Star className="h-3 w-3" />
            {repo.stargazers_count.toLocaleString()}
          </span>
          <span className="flex items-center gap-1">
            <GitFork className="h-3 w-3" />
            {repo.forks_count.toLocaleString()}
          </span>
          {repo.language && (
            <span className="flex items-center gap-1">
              <Code className="h-3 w-3" />
              {repo.language}
            </span>
          )}
        </div>

        <div className="flex gap-2 pt-2">
          <Button variant="outline" size="sm" asChild>
            <a href={repo.html_url} target="_blank" rel="noopener noreferrer">
              <ExternalLink className="mr-1 h-3 w-3" />
              View Repository
            </a>
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
