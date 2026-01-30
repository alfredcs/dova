import { useState } from 'react'
import { Plus, X, Tag } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import type { UserInterest } from '@/api/types'

interface InterestsEditorProps {
  interests: UserInterest[]
  onAdd: (topic: string) => void
  onRemove: (topic: string) => void
}

export default function InterestsEditor({
  interests,
  onAdd,
  onRemove,
}: InterestsEditorProps) {
  const [newTopic, setNewTopic] = useState('')

  const handleAdd = () => {
    if (newTopic.trim()) {
      onAdd(newTopic.trim())
      setNewTopic('')
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Tag className="h-5 w-5" />
          Research Interests
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2">
          <Input
            value={newTopic}
            onChange={(e) => setNewTopic(e.target.value)}
            placeholder="Add a topic..."
            onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
          />
          <Button onClick={handleAdd} size="icon">
            <Plus className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex flex-wrap gap-2">
          {interests.map((interest) => (
            <Badge
              key={interest.topic}
              variant="secondary"
              className="flex items-center gap-1 py-1"
            >
              {interest.topic}
              <button
                onClick={() => onRemove(interest.topic)}
                className="ml-1 rounded-full hover:bg-muted"
              >
                <X className="h-3 w-3" />
              </button>
            </Badge>
          ))}
          {interests.length === 0 && (
            <p className="text-sm text-muted-foreground">
              No interests added yet
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
