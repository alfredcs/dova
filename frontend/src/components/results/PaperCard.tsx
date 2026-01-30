import { FileText, Calendar, Users, ExternalLink } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { ArxivPaper } from '@/api/types'

interface PaperCardProps {
  paper: ArxivPaper
}

export default function PaperCard({ paper }: PaperCardProps) {
  const publishDate = new Date(paper.published).toLocaleDateString()

  return (
    <Card className="transition-shadow hover:shadow-md">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="line-clamp-2 text-base font-medium">
            {paper.title}
          </CardTitle>
          <Badge variant="arxiv">ArXiv</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="line-clamp-3 text-sm text-muted-foreground">
          {paper.abstract}
        </p>

        <div className="flex flex-wrap gap-1">
          {paper.categories.slice(0, 3).map((cat) => (
            <Badge key={cat} variant="outline" className="text-xs">
              {cat}
            </Badge>
          ))}
        </div>

        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <Users className="h-3 w-3" />
            {paper.authors.slice(0, 2).join(', ')}
            {paper.authors.length > 2 && ` +${paper.authors.length - 2}`}
          </span>
          <span className="flex items-center gap-1">
            <Calendar className="h-3 w-3" />
            {publishDate}
          </span>
        </div>

        <div className="flex gap-2 pt-2">
          <Button variant="outline" size="sm" asChild>
            <a href={paper.arxiv_url} target="_blank" rel="noopener noreferrer">
              <FileText className="mr-1 h-3 w-3" />
              View
            </a>
          </Button>
          <Button variant="outline" size="sm" asChild>
            <a href={paper.pdf_url} target="_blank" rel="noopener noreferrer">
              <ExternalLink className="mr-1 h-3 w-3" />
              PDF
            </a>
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
