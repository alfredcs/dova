import { Checkbox } from '@/components/ui/checkbox'

const sources = [
  { id: 'arxiv', label: 'ArXiv Papers', color: 'text-red-600' },
  { id: 'github', label: 'GitHub Repos', color: 'text-gray-700' },
  { id: 'huggingface', label: 'HuggingFace Models', color: 'text-yellow-600' },
]

interface SearchFiltersProps {
  selectedSources: string[]
  onToggle: (source: string) => void
}

export default function SearchFilters({
  selectedSources,
  onToggle,
}: SearchFiltersProps) {
  return (
    <div className="flex items-center gap-6">
      <span className="text-sm text-muted-foreground">Sources:</span>
      {sources.map((source) => (
        <label
          key={source.id}
          className="flex cursor-pointer items-center gap-2"
        >
          <Checkbox
            checked={selectedSources.includes(source.id)}
            onCheckedChange={() => onToggle(source.id)}
          />
          <span className={`text-sm ${source.color}`}>{source.label}</span>
        </label>
      ))}
    </div>
  )
}
