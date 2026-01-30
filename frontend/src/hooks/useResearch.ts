import { useMutation, useQuery } from '@tanstack/react-query'
import { conductResearch, getSearchHistory } from '@/api/research'
import type { ResearchQuery } from '@/api/types'

export function useResearch() {
  return useMutation({
    mutationFn: (query: ResearchQuery) => conductResearch(query),
  })
}

export function useSearchHistory() {
  return useQuery({
    queryKey: ['searchHistory'],
    queryFn: getSearchHistory,
  })
}
