import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Brain, Trash2, Star, Clock } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { getHistory, getKnowledge, deleteMemory } from '@/api/memory'

export default function Memory() {
  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState('history')

  const {
    data: history,
    isLoading: historyLoading,
    error: historyError,
  } = useQuery({
    queryKey: ['memory-history'],
    queryFn: getHistory,
  })

  const {
    data: knowledge,
    isLoading: knowledgeLoading,
    error: knowledgeError,
  } = useQuery({
    queryKey: ['memory-knowledge'],
    queryFn: getKnowledge,
  })

  const deleteMutation = useMutation({
    mutationFn: deleteMemory,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memory-history'] })
    },
  })

  if (activeTab === 'history' && historyLoading) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-bold">Memory</h1>
        {[...Array(5)].map((_, i) => (
          <Skeleton key={i} className="h-24" />
        ))}
      </div>
    )
  }

  if (activeTab === 'knowledge' && knowledgeLoading) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-bold">Memory</h1>
        {[...Array(3)].map((_, i) => (
          <Skeleton key={i} className="h-32" />
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Memory</h1>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="history">History</TabsTrigger>
          <TabsTrigger value="knowledge">Knowledge</TabsTrigger>
        </TabsList>

        <TabsContent value="history" className="mt-4 space-y-4">
          {historyError ? (
            <div className="flex h-64 items-center justify-center text-muted-foreground">
              Failed to load memory history. Please try again.
            </div>
          ) : history?.entries && history.entries.length > 0 ? (
            history.entries.map((entry) => (
              <Card key={entry.id} className="transition-shadow hover:shadow-md">
                <CardHeader className="flex flex-row items-center justify-between py-3">
                  <div className="flex items-center gap-3">
                    <Badge variant="outline">{entry.type.replace('_', ' ')}</Badge>
                    <span className="flex items-center gap-1 text-sm text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      {new Date(entry.created_at).toLocaleString()}
                    </span>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => deleteMutation.mutate(entry.id)}
                    disabled={deleteMutation.isPending}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </CardHeader>
                <CardContent className="py-2">
                  <p className="text-sm">
                    {entry.summary_text || JSON.stringify(entry.content)}
                  </p>
                </CardContent>
              </Card>
            ))
          ) : (
            <Card>
              <CardContent className="flex h-64 flex-col items-center justify-center text-muted-foreground">
                <Brain className="mb-4 h-12 w-12" />
                <p>No memory entries yet</p>
                <p className="text-sm">Your interactions will be remembered here</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="knowledge" className="mt-4 space-y-4">
          {knowledgeError ? (
            <div className="flex h-64 items-center justify-center text-muted-foreground">
              Failed to load knowledge. Please try again.
            </div>
          ) : knowledge && knowledge.length > 0 ? (
            knowledge.map((item) => (
              <Card key={item.id} className="transition-shadow hover:shadow-md">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Star className="h-4 w-4 text-yellow-500" />
                    {item.topic}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm">{item.summary}</p>
                  <p className="mt-2 text-xs text-muted-foreground">
                    Promoted: {new Date(item.promoted_at).toLocaleDateString()}
                  </p>
                </CardContent>
              </Card>
            ))
          ) : (
            <Card>
              <CardContent className="flex h-64 flex-col items-center justify-center text-muted-foreground">
                <Star className="mb-4 h-12 w-12" />
                <p>No knowledge items yet</p>
                <p className="text-sm">Crystallized insights will appear here</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}
