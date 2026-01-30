import { User, Calendar, Mail } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import { Badge } from '@/components/ui/badge'
import type { UserProfile } from '@/api/types'

interface ProfileCardProps {
  profile: UserProfile
}

export default function ProfileCard({ profile }: ProfileCardProps) {
  const initials = profile.name
    .split(' ')
    .map((n) => n[0])
    .join('')
    .toUpperCase()
    .slice(0, 2)

  const memberSince = new Date(profile.created_at).toLocaleDateString()

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <User className="h-5 w-5" />
          Profile
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-4">
          <Avatar className="h-16 w-16">
            <AvatarFallback className="text-lg">{initials}</AvatarFallback>
          </Avatar>
          <div>
            <h3 className="text-lg font-semibold">{profile.name}</h3>
            <p className="flex items-center gap-1 text-sm text-muted-foreground">
              <Mail className="h-3 w-3" />
              {profile.email}
            </p>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Expertise Level</span>
            <Badge variant="secondary" className="capitalize">
              {profile.expertise_level}
            </Badge>
          </div>

          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Member Since</span>
            <span className="flex items-center gap-1">
              <Calendar className="h-3 w-3" />
              {memberSince}
            </span>
          </div>
        </div>

        <div>
          <h4 className="mb-2 text-sm font-medium">Preferred Sources</h4>
          <div className="flex flex-wrap gap-1">
            {profile.preferred_sources.map((source) => (
              <Badge key={source} variant="outline" className="capitalize">
                {source}
              </Badge>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
