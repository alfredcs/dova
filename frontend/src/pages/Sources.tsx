import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Plus, Globe, Rss, Code, Trash2, Star, Terminal, RefreshCw, CheckCircle, XCircle, HelpCircle, Cloud } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { getSources, createSource, deleteSource, updateSource } from '@/api/sources'
import { getMCPServers } from '@/api/mcp'
import type { Source, CreateSourceRequest, MCPServer } from '@/api/types'

const sourceIcons = {
  builtin: Star,
  web_url: Globe,
  rss_feed: Rss,
  api: Code,
}

// Display names for MCP servers
const mcpDisplayNames: Record<string, string> = {
  arxiv: 'ArXiv Papers',
  github: 'GitHub',
  huggingface: 'HuggingFace',
  'hugging-face': 'HuggingFace',
  'awslabs.aws-documentation-mcp-server': 'AWS Documentation',
  'awslabs.aws-api-mcp-server': 'AWS API',
  'awslabs.core-mcp-server': 'AWS Core',
  'awslabs.eks-mcp-server': 'Amazon EKS',
  'awslabs.lambda-tool-mcp-server': 'AWS Lambda',
  'awslabs.aws-serverless-mcp-server': 'AWS Serverless',
  'awslabs.dynamodb-mcp-server': 'DynamoDB',
  'awslabs.bedrock-kb-retrieval-mcp-server': 'Bedrock Knowledge Base',
  'awslabs.cdk-mcp-server': 'AWS CDK',
  'awslabs.cloudwatch-mcp-server': 'CloudWatch',
  'awslabs.s3-tables-mcp-server': 'S3 Tables',
  'awslabs.iam-mcp-server': 'AWS IAM',
  'awslabs.terraform-mcp-server': 'Terraform',
  'awslabs.cfn-mcp-server': 'CloudFormation',
}

function getMCPDisplayName(name: string): string {
  if (mcpDisplayNames[name]) return mcpDisplayNames[name]
  // Auto-generate from name
  return name
    .replace('awslabs.', '')
    .replace(/-mcp-server$/, '')
    .split('-')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ')
}

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'healthy':
      return <CheckCircle className="h-4 w-4 text-green-500" />
    case 'unhealthy':
      return <XCircle className="h-4 w-4 text-red-500" />
    default:
      return <HelpCircle className="h-4 w-4 text-gray-400" />
  }
}

export default function Sources() {
  const queryClient = useQueryClient()
  const [showForm, setShowForm] = useState(false)
  const [formData, setFormData] = useState<CreateSourceRequest>({
    name: '',
    source_type: 'web_url',
    config: { url: '' },
  })

  // Custom sources query
  const { data: sources, isLoading: sourcesLoading, error: sourcesError } = useQuery({
    queryKey: ['sources'],
    queryFn: () => getSources(),
  })

  // MCP servers query
  const { data: mcpData, isLoading: mcpLoading, error: mcpError, refetch: refetchMcp } = useQuery({
    queryKey: ['mcp-servers-health'],
    queryFn: () => getMCPServers(true), // Check health
    staleTime: 60 * 1000, // 1 minute
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

  // Group MCP servers by category
  const mcpServers = mcpData?.servers || []
  const httpServers = mcpServers.filter(s => s.transport === 'http')
  const stdioServers = mcpServers.filter(s => s.transport === 'stdio')
  const awsServers = stdioServers.filter(s => s.name.startsWith('awslabs.'))
  const otherStdioServers = stdioServers.filter(s => !s.name.startsWith('awslabs.'))

  return (
    <div className="space-y-6">
      <Tabs defaultValue="mcp" className="w-full">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold">Sources</h1>
          <TabsList>
            <TabsTrigger value="mcp">MCP Servers</TabsTrigger>
            <TabsTrigger value="custom">Custom Sources</TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="mcp" className="space-y-6">
          <div className="flex items-center justify-between">
            <p className="text-muted-foreground">
              Model Context Protocol servers provide access to external data sources.
            </p>
            <Button
              variant="outline"
              size="sm"
              onClick={() => refetchMcp()}
              disabled={mcpLoading}
            >
              <RefreshCw className={`mr-2 h-4 w-4 ${mcpLoading ? 'animate-spin' : ''}`} />
              Refresh Status
            </Button>
          </div>

          {mcpLoading ? (
            <div className="space-y-4">
              {[...Array(4)].map((_, i) => (
                <Skeleton key={i} className="h-20" />
              ))}
            </div>
          ) : mcpError ? (
            <Card>
              <CardContent className="flex h-64 items-center justify-center text-muted-foreground">
                Failed to load MCP servers. Please try again.
              </CardContent>
            </Card>
          ) : (
            <>
              {/* HTTP Servers (Remote APIs) */}
              {httpServers.length > 0 && (
                <div className="space-y-4">
                  <h2 className="text-lg font-semibold flex items-center gap-2">
                    <Globe className="h-5 w-5" />
                    Remote API Servers
                  </h2>
                  <div className="grid gap-4 md:grid-cols-2">
                    {httpServers.map((server: MCPServer) => (
                      <Card key={server.name}>
                        <CardHeader className="flex flex-row items-center justify-between py-3">
                          <div className="flex items-center gap-3">
                            <StatusIcon status={server.status} />
                            <CardTitle className="text-base">{getMCPDisplayName(server.name)}</CardTitle>
                          </div>
                          <Badge variant="outline">HTTP</Badge>
                        </CardHeader>
                        <CardContent className="py-2 text-sm text-muted-foreground">
                          {server.url && <div className="truncate">{server.url}</div>}
                          {server.status_message && server.status !== 'healthy' && (
                            <div className="text-red-500 mt-1">{server.status_message}</div>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              )}

              {/* Local STDIO Servers (non-AWS) */}
              {otherStdioServers.length > 0 && (
                <div className="space-y-4">
                  <h2 className="text-lg font-semibold flex items-center gap-2">
                    <Terminal className="h-5 w-5" />
                    Local Servers
                  </h2>
                  <div className="grid gap-4 md:grid-cols-2">
                    {otherStdioServers.map((server: MCPServer) => (
                      <Card key={server.name}>
                        <CardHeader className="flex flex-row items-center justify-between py-3">
                          <div className="flex items-center gap-3">
                            <StatusIcon status={server.status} />
                            <CardTitle className="text-base">{getMCPDisplayName(server.name)}</CardTitle>
                          </div>
                          <Badge variant="secondary">STDIO</Badge>
                        </CardHeader>
                        <CardContent className="py-2 text-sm text-muted-foreground">
                          {server.command && (
                            <div className="font-mono text-xs truncate">{server.command}</div>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              )}

              {/* AWS MCP Servers */}
              {awsServers.length > 0 && (
                <div className="space-y-4">
                  <h2 className="text-lg font-semibold flex items-center gap-2">
                    <Cloud className="h-5 w-5 text-orange-500" />
                    AWS MCP Servers ({awsServers.length})
                  </h2>
                  <div className="grid gap-3 md:grid-cols-3 lg:grid-cols-4">
                    {awsServers.map((server: MCPServer) => (
                      <Card key={server.name} className="hover:border-orange-300 transition-colors">
                        <CardHeader className="py-3">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <StatusIcon status={server.status} />
                              <CardTitle className="text-sm font-medium">
                                {getMCPDisplayName(server.name)}
                              </CardTitle>
                            </div>
                          </div>
                        </CardHeader>
                        <CardContent className="py-2">
                          <p className="text-xs text-muted-foreground line-clamp-2" title={server.description}>
                            {server.description || server.name}
                          </p>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              )}

              {mcpServers.length === 0 && (
                <Card>
                  <CardContent className="flex h-64 flex-col items-center justify-center text-muted-foreground">
                    <Terminal className="mb-4 h-12 w-12" />
                    <p>No MCP servers configured</p>
                    <p className="text-sm">Add servers via ~/.dova.json or run dova mcp setup</p>
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </TabsContent>

        <TabsContent value="custom" className="space-y-6">
          <div className="flex items-center justify-between">
            <p className="text-muted-foreground">
              Add custom web sources for research.
            </p>
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

          {sourcesLoading ? (
            <div className="space-y-4">
              {[...Array(4)].map((_, i) => (
                <Skeleton key={i} className="h-24" />
              ))}
            </div>
          ) : sourcesError ? (
            <Card>
              <CardContent className="flex h-64 items-center justify-center text-muted-foreground">
                Failed to load sources. Please try again.
              </CardContent>
            </Card>
          ) : (
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
                    <p>No custom sources configured</p>
                    <p className="text-sm">Add custom sources to enhance your research</p>
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}
