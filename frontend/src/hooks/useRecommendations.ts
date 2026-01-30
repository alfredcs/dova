import { useQuery } from '@tanstack/react-query'
import { getRecommendations } from '@/api/profile'

export function useRecommendations() {
  return useQuery({
    queryKey: ['recommendations'],
    queryFn: getRecommendations,
    staleTime: 1000 * 60 * 10, // 10 minutes
  })
}
