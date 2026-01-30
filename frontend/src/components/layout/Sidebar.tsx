import { NavLink } from 'react-router-dom'
import { LayoutDashboard, User, History, Lightbulb, Brain, Database } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Checkbox } from '@/components/ui/checkbox'

const navItems = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/profile', icon: User, label: 'Profile' },
  { to: '/history', icon: History, label: 'History' },
  { to: '/memory', icon: Brain, label: 'Memory' },
  { to: '/sources', icon: Database, label: 'Sources' },
]

const sources = [
  { id: 'arxiv', label: 'ArXiv' },
  { id: 'github', label: 'GitHub' },
  { id: 'huggingface', label: 'HuggingFace' },
]

interface SidebarProps {
  selectedSources?: string[]
  onSourceToggle?: (source: string) => void
  recommendations?: { topic: string; id: string }[]
}

export default function Sidebar({
  selectedSources = ['arxiv', 'github', 'huggingface'],
  onSourceToggle,
  recommendations = [],
}: SidebarProps) {
  return (
    <aside className="flex w-64 flex-col border-r bg-card">
      {/* Logo */}
      <div className="flex h-16 items-center gap-2 border-b px-6">
        <img src="/dova-logo.svg" alt="DOVA" className="h-8 w-8" />
        <span className="text-xl font-bold text-primary">DOVA</span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 p-4">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
              )
            }
          >
            <item.icon className="h-4 w-4" />
            {item.label}
          </NavLink>
        ))}

        {/* Source Filters */}
        <div className="mt-6 border-t pt-4">
          <h3 className="mb-3 px-3 text-xs font-semibold uppercase text-muted-foreground">
            Sources
          </h3>
          <div className="space-y-2">
            {sources.map((source) => (
              <label
                key={source.id}
                className="flex cursor-pointer items-center gap-3 rounded-md px-3 py-2 text-sm hover:bg-accent"
              >
                <Checkbox
                  checked={selectedSources.includes(source.id)}
                  onCheckedChange={() => onSourceToggle?.(source.id)}
                />
                <span>{source.label}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Recommendations */}
        {recommendations.length > 0 && (
          <div className="mt-6 border-t pt-4">
            <h3 className="mb-3 flex items-center gap-2 px-3 text-xs font-semibold uppercase text-muted-foreground">
              <Lightbulb className="h-3 w-3" />
              Recommendations
            </h3>
            <div className="space-y-1">
              {recommendations.slice(0, 5).map((rec) => (
                <button
                  key={rec.id}
                  className="w-full truncate rounded-md px-3 py-2 text-left text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                >
                  {rec.topic}
                </button>
              ))}
            </div>
          </div>
        )}
      </nav>
    </aside>
  )
}
