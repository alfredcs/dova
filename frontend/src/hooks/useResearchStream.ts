import { useState, useCallback, useRef } from 'react'
import type { ResearchQuery, ResearchResponse, TransactionLogEntry } from '@/api/types'

type StreamStatus = 'idle' | 'streaming' | 'complete' | 'error'
type Stage = 'deliberating' | 'searching' | 'synthesizing' | null

export interface PartialResults {
  papers: Record<string, unknown>[]
  repositories: Record<string, unknown>[]
  models: Record<string, unknown>[]
  datasets: Record<string, unknown>[]
  web_results: Record<string, unknown>[]
  images: Record<string, unknown>[]
}

const emptyPartial: PartialResults = {
  papers: [],
  repositories: [],
  models: [],
  datasets: [],
  web_results: [],
  images: [],
}

function hasAnyResults(p: PartialResults): boolean {
  return (
    p.papers.length > 0 ||
    p.repositories.length > 0 ||
    p.models.length > 0 ||
    p.datasets.length > 0 ||
    p.web_results.length > 0 ||
    p.images.length > 0
  )
}

export function useResearchStream() {
  const [status, setStatus] = useState<StreamStatus>('idle')
  const [currentStage, setCurrentStage] = useState<Stage>(null)
  const [stageMessage, setStageMessage] = useState<string>('')
  const [partialResults, setPartialResults] = useState<PartialResults>(emptyPartial)
  const [transactionLog, setTransactionLog] = useState<TransactionLogEntry[]>([])
  const [finalResult, setFinalResult] = useState<ResearchResponse | undefined>()
  const [streamingAnswer, setStreamingAnswer] = useState<string>('')
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const search = useCallback(async (query: ResearchQuery) => {
    // Reset state
    setStatus('streaming')
    setCurrentStage(null)
    setStageMessage('')
    setPartialResults(emptyPartial)
    setTransactionLog([])
    setFinalResult(undefined)
    setStreamingAnswer('')
    setError(null)

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    const token = localStorage.getItem('auth_token')
    const sessionId = sessionStorage.getItem('session_id') || crypto.randomUUID()
    sessionStorage.setItem('session_id', sessionId)

    const baseUrl = import.meta.env.VITE_API_URL || '/api'

    // Accumulate synthesis tokens in a local buffer, flush to state
    // periodically to avoid hundreds of re-renders per second.
    let answerBuffer = ''
    let flushScheduled = false

    function flushAnswer() {
      flushScheduled = false
      const snapshot = answerBuffer
      setStreamingAnswer(snapshot)
    }

    function handleEvent(event: string, data: Record<string, unknown>) {
      switch (event) {
        case 'stage':
          setCurrentStage(data.stage as Stage)
          if (data.message) setStageMessage(data.message as string)
          break

        case 'tool_complete': {
          const resultKey = data.result_key as string
          const items = (data.items || []) as Record<string, unknown>[]
          if (resultKey === 'huggingface') {
            setPartialResults((prev) => ({
              ...prev,
              models: [...prev.models, ...items],
            }))
          } else if (resultKey in emptyPartial) {
            setPartialResults((prev) => ({
              ...prev,
              [resultKey]: [...(prev[resultKey as keyof PartialResults] || []), ...items],
            }))
          }
          break
        }

        case 'synthesis_token':
          answerBuffer += data.token as string
          if (!flushScheduled) {
            flushScheduled = true
            requestAnimationFrame(flushAnswer)
          }
          break

        case 'log':
          setTransactionLog((prev) => [...prev, data as unknown as TransactionLogEntry])
          break

        case 'complete':
          // Final flush of any remaining buffered tokens
          setStreamingAnswer('')
          setFinalResult(data as unknown as ResearchResponse)
          setStatus('complete')
          break

        case 'error':
          setError(data.message as string)
          setStatus('error')
          break
      }
    }

    try {
      let response: Response

      if (query.files && query.files.length > 0) {
        const formData = new FormData()
        formData.append('query', query.query)
        if (query.sources) formData.append('sources', JSON.stringify(query.sources))
        if (query.max_results !== undefined) formData.append('max_results', String(query.max_results))
        if (query.orchestrator) formData.append('orchestrator', query.orchestrator)
        for (const file of query.files) formData.append('files', file)

        response = await fetch(`${baseUrl}/v1/research/stream/upload`, {
          method: 'POST',
          headers: {
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
            'X-Session-ID': sessionId,
          },
          body: formData,
          signal: controller.signal,
        })
      } else {
        response = await fetch(`${baseUrl}/v1/research/stream`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
            'X-Session-ID': sessionId,
          },
          body: JSON.stringify({
            query: query.query,
            sources: query.sources,
            max_results: query.max_results,
            orchestrator: query.orchestrator || 'thinking',
          }),
          signal: controller.signal,
        })
      }

      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || `HTTP ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error('No response body')

      const decoder = new TextDecoder()
      let buffer = ''
      let currentEvent = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // Parse complete SSE events from buffer.
        // An SSE event ends with a blank line (\n\n). We only process
        // up to the last blank-line boundary and keep the rest in the buffer.
        const boundary = buffer.lastIndexOf('\n\n')
        if (boundary === -1) continue // no complete event yet
        const complete = buffer.slice(0, boundary + 2)
        buffer = buffer.slice(boundary + 2)

        const lines = complete.split('\n')
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim()
          } else if (line.startsWith('data: ') && currentEvent) {
            try {
              const data = JSON.parse(line.slice(6))
              handleEvent(currentEvent, data)
            } catch {
              // skip malformed JSON
            }
            currentEvent = ''
          }
        }
      }

      // Final flush in case any tokens remain unbatched
      if (answerBuffer) flushAnswer()

      // If we didn't get a 'complete' event, mark as complete anyway
      setStatus((prev) => (prev === 'streaming' ? 'complete' : prev))
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') return
      const msg = err instanceof Error ? err.message : 'Unknown error'
      setError(msg)
      setStatus('error')
    }
  }, [])

  const abort = useCallback(() => {
    abortRef.current?.abort()
    setStatus('idle')
  }, [])

  return {
    search,
    abort,
    status,
    currentStage,
    stageMessage,
    partialResults,
    transactionLog,
    finalResult,
    streamingAnswer,
    error,
    isStreaming: status === 'streaming',
    hasResults: hasAnyResults(partialResults) || finalResult !== undefined,
  }
}
