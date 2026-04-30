import { Checkbox } from '@/components/ui/checkbox'

// Three top-level source groups — these IDs are what the UI stores in
// selection state and what downstream code toggles. Before hitting the
// API, `expandSourceGroups` maps them to concrete backend source names
// (AI → arxiv/github/huggingface). The orchestrator then performs
// deliberation-first selection across the expanded set and semantic
// fan-out within groups like `bio`.
export type SourceGroupId = 'ai' | 'web' | 'bio'

export const SOURCE_GROUPS: {
  id: SourceGroupId
  label: string
  color: string
  description: string
  sources: string[]
}[] = [
  {
    id: 'ai',
    label: 'AI',
    color: 'text-indigo-600',
    description:
      'AI / ML research: ArXiv papers, GitHub repos, HuggingFace models and datasets',
    sources: ['arxiv', 'github', 'huggingface'],
  },
  {
    id: 'web',
    label: 'Web',
    color: 'text-blue-600',
    description: 'General web search across Brave, Perplexity, Tavily, DuckDuckGo',
    sources: ['web'],
  },
  {
    id: 'bio',
    label: 'Bio',
    color: 'text-emerald-600',
    description:
      'Biotech / pharma: PubMed literature, ClinicalTrials.gov, PubChem compounds (routed by the orchestrator)',
    sources: ['bio'],
  },
]

/**
 * Expand group IDs ('ai', 'web', 'bio') into the concrete backend source
 * names the API and orchestrator understand. Unknown IDs pass through.
 */
export function expandSourceGroups(groupIds: string[]): string[] {
  const out = new Set<string>()
  for (const id of groupIds) {
    const group = SOURCE_GROUPS.find((g) => g.id === id)
    if (group) {
      group.sources.forEach((s) => out.add(s))
    } else {
      out.add(id)
    }
  }
  return Array.from(out)
}

const orchestrators = [
  { id: 'standard', label: 'Standard', description: 'Task-graph orchestration' },
  { id: 'thinking', label: 'Thinking', description: 'Deliberation-first (experimental)' },
]

interface SearchFiltersProps {
  /** Currently-selected group IDs (e.g. ['ai', 'web', 'bio']). */
  selectedSources: string[]
  /** Called with the group ID being toggled. */
  onToggle: (groupId: string) => void
  orchestrator?: 'standard' | 'thinking'
  onOrchestratorChange?: (orchestrator: 'standard' | 'thinking') => void
}

export default function SearchFilters({
  selectedSources,
  onToggle,
  orchestrator = 'thinking',
  onOrchestratorChange,
}: SearchFiltersProps) {
  return (
    <div className="flex flex-wrap items-center gap-6">
      <div className="flex flex-wrap items-center gap-4">
        <span className="text-sm text-muted-foreground">Sources:</span>
        {SOURCE_GROUPS.map((group) => (
          <label
            key={group.id}
            className="flex cursor-pointer items-center gap-2"
            title={group.description}
          >
            <Checkbox
              checked={selectedSources.includes(group.id)}
              onCheckedChange={() => onToggle(group.id)}
            />
            <span className={`text-sm font-medium ${group.color}`}>{group.label}</span>
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
