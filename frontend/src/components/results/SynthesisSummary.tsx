import { Sparkles, CheckCircle, ArrowRight } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { SynthesisSummary as SynthesisSummaryType } from '@/api/types'

interface SynthesisSummaryProps {
  synthesis: SynthesisSummaryType
}

export default function SynthesisSummary({ synthesis }: SynthesisSummaryProps) {
  return (
    <Card className="border-primary/20 bg-gradient-to-r from-primary/5 to-transparent">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Sparkles className="h-5 w-5 text-primary" />
          AI Synthesis
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">{synthesis.summary}</p>

        {synthesis.key_findings.length > 0 && (
          <div>
            <h4 className="mb-2 text-sm font-medium">Key Findings</h4>
            <ul className="space-y-1">
              {synthesis.key_findings.map((finding, i) => (
                <li key={i} className="flex items-start gap-2 text-sm">
                  <CheckCircle className="mt-0.5 h-4 w-4 shrink-0 text-green-500" />
                  <span>{finding}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {synthesis.recommended_actions.length > 0 && (
          <div>
            <h4 className="mb-2 text-sm font-medium">Recommended Actions</h4>
            <ul className="space-y-1">
              {synthesis.recommended_actions.map((action, i) => (
                <li key={i} className="flex items-start gap-2 text-sm">
                  <ArrowRight className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                  <span>{action}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
