import { useState, useRef, KeyboardEvent } from 'react'
import { Search, Loader2, Paperclip, X } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'

interface SearchBarProps {
  onSearch: (query: string, files: File[]) => void
  isLoading?: boolean
  placeholder?: string
}

export default function SearchBar({
  onSearch,
  isLoading = false,
  placeholder = 'Search papers, repositories, and models...',
}: SearchBarProps) {
  const [query, setQuery] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleSearch = () => {
    if (query.trim()) {
      onSearch(query.trim(), files)
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSearch()
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files)
      setFiles((prev) => [...prev, ...newFiles].slice(0, 5))
    }
    // Reset input so the same file can be re-selected
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  return (
    <div className="flex w-full max-w-3xl flex-col gap-2">
      <div className="flex gap-2">
        <div className="relative flex-1">
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isLoading}
            className="absolute left-3 top-1/2 z-10 -translate-y-1/2 text-muted-foreground hover:text-foreground disabled:opacity-50"
            title="Attach files (.txt, .pdf, .png, .jpg, .csv, .md, .tex)"
          >
            <Paperclip className="h-4 w-4" />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.pdf,.png,.jpg,.csv,.md,.tex"
            multiple
            className="hidden"
            onChange={handleFileChange}
          />
          <Search className="absolute left-9 top-1/2 z-10 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className="pl-16"
            disabled={isLoading}
          />
        </div>
        <Button onClick={handleSearch} disabled={isLoading || !query.trim()}>
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Searching...
            </>
          ) : (
            'Search'
          )}
        </Button>
      </div>
      {files.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {files.map((file, index) => (
            <span
              key={`${file.name}-${index}`}
              className="inline-flex items-center gap-1 rounded-full bg-muted px-3 py-1 text-sm"
            >
              {file.name}
              <button
                type="button"
                onClick={() => removeFile(index)}
                className="ml-1 text-muted-foreground hover:text-foreground"
              >
                <X className="h-3 w-3" />
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
