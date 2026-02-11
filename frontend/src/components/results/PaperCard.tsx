import { FileText, Calendar, Users, ExternalLink } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'

interface PaperCardProps {
  paper: Record<string, unknown>
}

export default function PaperCard({ paper }: PaperCardProps) {
  const title = (paper.title as string) || 'Untitled'
  const abstract = (paper.abstract as string) || (paper.description as string) || ''
  const authors: string[] = Array.isArray(paper.authors) ? paper.authors.map(String) : []
  const categories: string[] = Array.isArray(paper.categories) ? paper.categories.map(String) : []
  const published = paper.published as string | undefined
  const arxivUrl = (paper.arxiv_url as string) || (paper.url as string) || ''
  const pdfUrl = (paper.pdf_url as string) || (arxivUrl && arxivUrl.includes('arxiv.org') ? arxivUrl.replace('/abs/', '/pdf/') : '')

  const publishDate = published ? new Date(published).toLocaleDateString() : ''

  return (
    <Card className="transition-shadow hover:shadow-md">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="line-clamp-2 text-base font-medium">
            {title}
          </CardTitle>
          <Badge variant="arxiv">ArXiv</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="line-clamp-3 text-sm text-muted-foreground">
          {abstract}
        </p>

        {categories.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {categories.slice(0, 3).map((cat) => (
              <Badge key={cat} variant="outline" className="text-xs">
                {cat}
              </Badge>
            ))}
          </div>
        )}

        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          {authors.length > 0 && (
            <span className="flex items-center gap-1">
              <Users className="h-3 w-3" />
              {authors.slice(0, 2).join(', ')}
              {authors.length > 2 && ` +${authors.length - 2}`}
            </span>
          )}
          {publishDate && (
            <span className="flex items-center gap-1">
              <Calendar className="h-3 w-3" />
              {publishDate}
            </span>
          )}
        </div>

        <div className="flex gap-2 pt-2">
          {arxivUrl && (
            <Button variant="outline" size="sm" asChild>
              <a href={arxivUrl} target="_blank" rel="noopener noreferrer">
                <FileText className="mr-1 h-3 w-3" />
                View
              </a>
            </Button>
          )}
          {pdfUrl && (
            <Button variant="outline" size="sm" asChild>
              <a href={pdfUrl} target="_blank" rel="noopener noreferrer">
                <ExternalLink className="mr-1 h-3 w-3" />
                PDF
              </a>
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
