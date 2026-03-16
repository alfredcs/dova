import { useState } from 'react'
import type { TransactionLogEntry } from '@/api/types'

interface ProgressTimelineProps {
  log: TransactionLogEntry[]
  stageMessage: string
  isStreaming: boolean
}

function StatusIcon({ status }: { status: string }) {
  if (status === 'started') {
    return (
      <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
    )
  }
  if (status === 'completed') {
    return (
      <div className="flex h-4 w-4 items-center justify-center rounded-full bg-green-500 text-white text-[10px]">
        ✓
      </div>
    )
  }
  // error
  return (
    <div className="flex h-4 w-4 items-center justify-center rounded-full bg-red-500 text-white text-[10px]">
      ✗
    </div>
  )
}

function formatStep(step: string): string {
  return step
    .replace(/_search$/, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

export default function ProgressTimeline({
  log,
  stageMessage,
  isStreaming,
}: ProgressTimelineProps) {
  const [collapsed, setCollapsed] = useState(false)

  if (log.length === 0 && !isStreaming) return null

  // Deduplicate: group by step, show latest status
  const stepMap = new Map<string, TransactionLogEntry>()
  for (const entry of log) {
    stepMap.set(entry.step, entry)
  }
  const steps = Array.from(stepMap.values())

  return (
    <div className="rounded-lg border bg-card p-4">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex w-full items-center justify-between text-sm font-medium text-muted-foreground hover:text-foreground"
      >
        <span className="flex items-center gap-2">
          {isStreaming && (
            <div className="h-2 w-2 animate-pulse rounded-full bg-blue-500" />
          )}
          {stageMessage || (isStreaming ? 'Processing...' : 'Research complete')}
        </span>
        <span className="text-xs">{collapsed ? '▸' : '▾'}</span>
      </button>

      {!collapsed && steps.length > 0 && (
        <div className="mt-3 space-y-2 border-l-2 border-muted pl-4">
          {steps.map((entry) => (
            <div key={entry.step} className="flex items-center gap-3 text-sm">
              <StatusIcon status={entry.status} />
              <span className="font-medium">{formatStep(entry.step)}</span>
              {entry.elapsed_ms !== undefined && entry.status !== 'started' && (
                <span className="text-xs text-muted-foreground">
                  {entry.elapsed_ms < 1000
                    ? `${entry.elapsed_ms}ms`
                    : `${(entry.elapsed_ms / 1000).toFixed(1)}s`}
                </span>
              )}
              {entry.detail && entry.status === 'error' && (
                <span className="text-xs text-red-500 truncate max-w-[200px]">
                  {entry.detail}
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
