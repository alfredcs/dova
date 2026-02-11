import Markdown from 'react-markdown'
import { Sparkles, CheckCircle, ArrowRight } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

interface SynthesisSummaryProps {
  answer: string;
  summary: string;
  insights: string[];
  recommendations: string[];
  confidence: number;
}

export default function SynthesisSummary({
  answer,
  summary,
  insights,
  recommendations,
  confidence,
}: SynthesisSummaryProps) {
  return (
    <Card className="border-primary/20 bg-gradient-to-r from-primary/5 to-transparent">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Sparkles className="h-5 w-5 text-primary" />
          AI Synthesis
          {confidence > 0 && (
            <span className="ml-auto text-sm font-normal text-muted-foreground">
              Confidence: {Math.round(confidence * 100)}%
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {answer && (
          <div className="prose prose-sm max-w-none dark:prose-invert">
            <Markdown>{answer}</Markdown>
          </div>
        )}

        {!answer && summary && (
          <p className="text-sm text-muted-foreground">{summary}</p>
        )}

        {insights.length > 0 && (
          <div>
            <h4 className="mb-2 text-sm font-medium">Key Insights</h4>
            <ul className="space-y-1">
              {insights.map((insight, i) => (
                <li key={i} className="flex items-start gap-2 text-sm">
                  <CheckCircle className="mt-0.5 h-4 w-4 shrink-0 text-green-500" />
                  <span>{insight}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {recommendations.length > 0 && (
          <div>
            <h4 className="mb-2 text-sm font-medium">Recommendations</h4>
            <ul className="space-y-1">
              {recommendations.map((rec, i) => (
                <li key={i} className="flex items-start gap-2 text-sm">
                  <ArrowRight className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
