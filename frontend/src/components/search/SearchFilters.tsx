import { Checkbox } from '@/components/ui/checkbox'

const sources = [
  { id: 'arxiv', label: 'ArXiv Papers', color: 'text-red-600' },
  { id: 'github', label: 'GitHub Repos', color: 'text-gray-700' },
  { id: 'huggingface', label: 'HuggingFace Models', color: 'text-yellow-600' },
]

const orchestrators = [
  { id: 'standard', label: 'Standard', description: 'Task-graph orchestration' },
  { id: 'thinking', label: 'Thinking', description: 'Deliberation-first (experimental)' },
]

interface SearchFiltersProps {
  selectedSources: string[]
  onToggle: (source: string) => void
  orchestrator?: 'standard' | 'thinking'
  onOrchestratorChange?: (orchestrator: 'standard' | 'thinking') => void
}

export default function SearchFilters({
  selectedSources,
  onToggle,
  orchestrator = 'standard',
  onOrchestratorChange,
}: SearchFiltersProps) {
  return (
    <div className="flex flex-wrap items-center gap-6">
      <div className="flex items-center gap-4">
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
      {onOrchestratorChange && (
        <div className="flex items-center gap-4 border-l pl-6">
          <span className="text-sm text-muted-foreground">Mode:</span>
          {orchestrators.map((orch) => (
            <label
              key={orch.id}
              className="flex cursor-pointer items-center gap-2"
              title={orch.description}
            >
              <Checkbox
                checked={orchestrator === orch.id}
                onCheckedChange={() =>
                  onOrchestratorChange(orch.id as 'standard' | 'thinking')
                }
              />
              <span className="text-sm">{orch.label}</span>
            </label>
          ))}
        </div>
      )}
    </div>
  )
}
