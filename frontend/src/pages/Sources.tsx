import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Plus, Globe, Rss, Code, Trash2, Star } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { getSources, createSource, deleteSource, updateSource } from '@/api/sources'
import type { Source, CreateSourceRequest } from '@/api/types'

const sourceIcons = {
  builtin: Star,
  web_url: Globe,
  rss_feed: Rss,
  api: Code,
}

export default function Sources() {
  const queryClient = useQueryClient()
  const [showForm, setShowForm] = useState(false)
  const [formData, setFormData] = useState<CreateSourceRequest>({
    name: '',
    source_type: 'web_url',
    config: { url: '' },
  })

  const { data: sources, isLoading, error } = useQuery({
    queryKey: ['sources'],
    queryFn: () => getSources(),
  })

  const createMutation = useMutation({
    mutationFn: createSource,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sources'] })
      setShowForm(false)
      setFormData({ name: '', source_type: 'web_url', config: { url: '' } })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: deleteSource,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['sources'] }),
  })

  const toggleMutation = useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) =>
      updateSource(id, { enabled }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['sources'] }),
  })

  if (isLoading) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-bold">Sources</h1>
        {[...Array(4)].map((_, i) => (
          <Skeleton key={i} className="h-24" />
        ))}
      </div>
    )
  }

  if (error) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-bold">Sources</h1>
        <div className="flex h-64 items-center justify-center text-muted-foreground">
          Failed to load sources. Please try again.
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Sources</h1>
        <Button onClick={() => setShowForm(!showForm)}>
          <Plus className="mr-2 h-4 w-4" /> Add Source
        </Button>
      </div>

      {showForm && (
        <Card>
          <CardContent className="space-y-4 pt-4">
            <Input
              placeholder="Source name"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
            />
            <select
              className="w-full rounded border p-2"
              value={formData.source_type}
              onChange={(e) =>
                setFormData({ ...formData, source_type: e.target.value as 'web_url' | 'rss_feed' | 'api' })
              }
            >
              <option value="web_url">Web URL</option>
              <option value="rss_feed">RSS Feed</option>
              <option value="api">API Endpoint</option>
            </select>
            <Input
              placeholder="URL (use {query} for search term)"
              value={formData.config.url}
              onChange={(e) =>
                setFormData({ ...formData, config: { ...formData.config, url: e.target.value } })
              }
            />
            <div className="flex gap-2">
              <Button
                onClick={() => createMutation.mutate(formData)}
                disabled={createMutation.isPending || !formData.name || !formData.config.url}
              >
                Create Source
              </Button>
              <Button variant="outline" onClick={() => setShowForm(false)}>
                Cancel
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="space-y-4">
        {sources && sources.length > 0 ? (
          sources.map((source: Source) => {
            const Icon = sourceIcons[source.source_type]
            return (
              <Card key={source.id} className={!source.enabled ? 'opacity-50' : ''}>
                <CardHeader className="flex flex-row items-center justify-between py-3">
                  <div className="flex items-center gap-3">
                    <Icon className="h-5 w-5" />
                    <CardTitle className="text-base">{source.name}</CardTitle>
                    <Badge variant="outline">{source.source_type.replace('_', ' ')}</Badge>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-muted-foreground">
                      Quality: {(source.quality.quality_score * 100).toFixed(0)}%
                    </span>
                    {source.source_type !== 'builtin' && (
                      <>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() =>
                            toggleMutation.mutate({ id: source.id, enabled: !source.enabled })
                          }
                          disabled={toggleMutation.isPending}
                        >
                          {source.enabled ? 'Disable' : 'Enable'}
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => deleteMutation.mutate(source.id)}
                          disabled={deleteMutation.isPending}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </>
                    )}
                  </div>
                </CardHeader>
                <CardContent className="py-2 text-sm text-muted-foreground">
                  {source.quality.query_count} queries · {source.quality.click_count} clicks
                </CardContent>
              </Card>
            )
          })
        ) : (
          <Card>
            <CardContent className="flex h-64 flex-col items-center justify-center text-muted-foreground">
              <Globe className="mb-4 h-12 w-12" />
              <p>No sources configured</p>
              <p className="text-sm">Add custom sources to enhance your research</p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
