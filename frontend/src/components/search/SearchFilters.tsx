import { useQuery } from '@tanstack/react-query'
import { Checkbox } from '@/components/ui/checkbox'
import { getMCPServers } from '@/api/mcp'

// Core sources that are always shown (with or without API)
const coreSources = [
  { id: 'arxiv', label: 'ArXiv Papers', color: 'text-red-600' },
  { id: 'github', label: 'GitHub Repos', color: 'text-gray-700' },
  { id: 'huggingface', label: 'HuggingFace', color: 'text-yellow-600' },
  { id: 'web', label: 'Web Search', color: 'text-blue-600' },
  { id: 'awslabs.aws-documentation-mcp-server', label: 'AWS Docs', color: 'text-orange-600' },
  { id: 'awslabs.bedrock-kb-retrieval-mcp-server', label: 'Bedrock KB', color: 'text-purple-600' },
]

// Aliases to handle different naming conventions
const sourceAliases: Record<string, string> = {
  'hugging-face': 'huggingface',
}

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
  orchestrator = 'thinking',
  onOrchestratorChange,
}: SearchFiltersProps) {
  // Optionally fetch API to check which servers are actually available
  // But we always show core sources regardless
  const { data } = useQuery({
    queryKey: ['mcp-servers'],
    queryFn: () => getMCPServers(false),
    staleTime: 5 * 60 * 1000,
    retry: false, // Don't retry on failure
  })

  // Build the list of sources to display
  // Start with core sources, then add any additional from API
  const coreIds = new Set(coreSources.map(s => s.id))
  const additionalSources = data?.servers
    .filter(s => s.enabled && !coreIds.has(s.name) && !coreIds.has(sourceAliases[s.name] || ''))
    .slice(0, 4) // Limit additional sources shown
    .map(s => ({
      id: s.name,
      label: s.name.replace('awslabs.', '').replace(/-mcp-server$/, '').split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
      color: s.name.startsWith('awslabs.') ? 'text-orange-500' : 'text-gray-600',
    })) || []

  const allSources = [...coreSources, ...additionalSources]

  // Check if a source is selected (handle aliases)
  const isSelected = (sourceId: string) => {
    if (selectedSources.includes(sourceId)) return true
    const alias = sourceAliases[sourceId]
    if (alias && selectedSources.includes(alias)) return true
    // Check reverse alias
    for (const [key, val] of Object.entries(sourceAliases)) {
      if (val === sourceId && selectedSources.includes(key)) return true
    }
    return false
  }

  return (
    <div className="flex flex-wrap items-center gap-6">
      <div className="flex flex-wrap items-center gap-4">
        <span className="text-sm text-muted-foreground">Sources:</span>
        {allSources.map((source) => (
          <label
            key={source.id}
            className="flex cursor-pointer items-center gap-2"
            title={source.id}
          >
            <Checkbox
              checked={isSelected(source.id)}
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
