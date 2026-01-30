import { Settings } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Checkbox } from '@/components/ui/checkbox'
import { Button } from '@/components/ui/button'

interface PreferencesFormProps {
  expertiseLevel: 'beginner' | 'intermediate' | 'expert'
  preferredSources: string[]
  onExpertiseChange: (level: 'beginner' | 'intermediate' | 'expert') => void
  onSourceToggle: (source: string) => void
  onSave: () => void
  isSaving?: boolean
}

const expertiseLevels = [
  { value: 'beginner', label: 'Beginner', description: 'New to the field' },
  { value: 'intermediate', label: 'Intermediate', description: 'Some experience' },
  { value: 'expert', label: 'Expert', description: 'Deep knowledge' },
] as const

const sources = [
  { id: 'arxiv', label: 'ArXiv Papers' },
  { id: 'github', label: 'GitHub Repositories' },
  { id: 'huggingface', label: 'HuggingFace Models' },
]

export default function PreferencesForm({
  expertiseLevel,
  preferredSources,
  onExpertiseChange,
  onSourceToggle,
  onSave,
  isSaving,
}: PreferencesFormProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings className="h-5 w-5" />
          Preferences
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div>
          <h4 className="mb-3 text-sm font-medium">Expertise Level</h4>
          <div className="space-y-2">
            {expertiseLevels.map((level) => (
              <label
                key={level.value}
                className="flex cursor-pointer items-center gap-3 rounded-md border p-3 hover:bg-accent"
              >
                <input
                  type="radio"
                  name="expertise"
                  value={level.value}
                  checked={expertiseLevel === level.value}
                  onChange={() => onExpertiseChange(level.value)}
                  className="h-4 w-4 text-primary"
                />
                <div>
                  <div className="text-sm font-medium">{level.label}</div>
                  <div className="text-xs text-muted-foreground">
                    {level.description}
                  </div>
                </div>
              </label>
            ))}
          </div>
        </div>

        <div>
          <h4 className="mb-3 text-sm font-medium">Preferred Sources</h4>
          <div className="space-y-2">
            {sources.map((source) => (
              <label
                key={source.id}
                className="flex cursor-pointer items-center gap-3 rounded-md border p-3 hover:bg-accent"
              >
                <Checkbox
                  checked={preferredSources.includes(source.id)}
                  onCheckedChange={() => onSourceToggle(source.id)}
                />
                <span className="text-sm">{source.label}</span>
              </label>
            ))}
          </div>
        </div>

        <Button onClick={onSave} disabled={isSaving} className="w-full">
          {isSaving ? 'Saving...' : 'Save Preferences'}
        </Button>
      </CardContent>
    </Card>
  )
}
