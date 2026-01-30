import { useState, useEffect } from 'react'
import ProfileCard from '@/components/profile/ProfileCard'
import InterestsEditor from '@/components/profile/InterestsEditor'
import PreferencesForm from '@/components/profile/PreferencesForm'
import { Skeleton } from '@/components/ui/skeleton'
import { useProfile, useUpdateProfile } from '@/hooks/useProfile'
import type { UserInterest, UserProfile } from '@/api/types'

export default function Profile() {
  const { data: profile, isLoading, error } = useProfile()
  const { mutate: updateProfile, isPending: isSaving } = useUpdateProfile()

  const [interests, setInterests] = useState<UserInterest[]>([])
  const [expertiseLevel, setExpertiseLevel] = useState<UserProfile['expertise_level']>('intermediate')
  const [preferredSources, setPreferredSources] = useState<string[]>([])

  useEffect(() => {
    if (profile) {
      setInterests(profile.interests)
      setExpertiseLevel(profile.expertise_level)
      setPreferredSources(profile.preferred_sources)
    }
  }, [profile])

  const handleAddInterest = (topic: string) => {
    const newInterest: UserInterest = {
      topic,
      weight: 1,
      added_at: new Date().toISOString(),
    }
    setInterests((prev) => [...prev, newInterest])
  }

  const handleRemoveInterest = (topic: string) => {
    setInterests((prev) => prev.filter((i) => i.topic !== topic))
  }

  const handleSourceToggle = (source: string) => {
    setPreferredSources((prev) =>
      prev.includes(source)
        ? prev.filter((s) => s !== source)
        : [...prev, source]
    )
  }

  const handleSave = () => {
    updateProfile({
      interests,
      expertise_level: expertiseLevel,
      preferred_sources: preferredSources,
    })
  }

  if (isLoading) {
    return (
      <div className="grid gap-6 md:grid-cols-2">
        <Skeleton className="h-64" />
        <Skeleton className="h-64" />
        <Skeleton className="h-80 md:col-span-2" />
      </div>
    )
  }

  if (error || !profile) {
    return (
      <div className="flex h-64 items-center justify-center text-muted-foreground">
        Failed to load profile. Please try again.
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Profile Settings</h1>

      <div className="grid gap-6 md:grid-cols-2">
        <ProfileCard profile={profile} />
        <PreferencesForm
          expertiseLevel={expertiseLevel}
          preferredSources={preferredSources}
          onExpertiseChange={setExpertiseLevel}
          onSourceToggle={handleSourceToggle}
          onSave={handleSave}
          isSaving={isSaving}
        />
      </div>

      <InterestsEditor
        interests={interests}
        onAdd={handleAddInterest}
        onRemove={handleRemoveInterest}
      />
    </div>
  )
}
