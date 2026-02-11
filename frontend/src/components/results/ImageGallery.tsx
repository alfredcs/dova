import { Download, Image as ImageIcon, Copy, Check } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useState, useMemo } from 'react'
import type { ImageResult } from '@/api/types'

interface ImageGalleryProps {
  images: ImageResult[]
}

/**
 * Convert image URL/data to a displayable src.
 * Handles: URLs, data URIs, and base64 encoded strings.
 */
function getImageSrc(url: string): string {
  if (!url) return ''

  // Already a valid URL or data URI
  if (url.startsWith('http://') || url.startsWith('https://') || url.startsWith('data:')) {
    return url
  }

  // Check if it's a file path (from Gradio spaces)
  if (url.startsWith('/') || url.includes('/tmp/') || url.includes('gradio')) {
    // This is a server-side path, can't display directly
    // Try to treat as base64 if it looks like one
    if (url.length > 100 && !url.includes('/')) {
      return `data:image/png;base64,${url}`
    }
    return url
  }

  // Assume it's base64 encoded - add data URI prefix
  // Try to detect image type from base64 header
  if (url.startsWith('iVBOR')) {
    // PNG signature in base64
    return `data:image/png;base64,${url}`
  } else if (url.startsWith('/9j/')) {
    // JPEG signature in base64
    return `data:image/jpeg;base64,${url}`
  } else if (url.startsWith('R0lGOD')) {
    // GIF signature in base64
    return `data:image/gif;base64,${url}`
  } else if (url.startsWith('UklGR')) {
    // WebP signature in base64
    return `data:image/webp;base64,${url}`
  }

  // Default to PNG for unknown base64
  return `data:image/png;base64,${url}`
}

function ImageCard({ image }: { image: ImageResult }) {
  const [copied, setCopied] = useState(false)
  const [imageError, setImageError] = useState(false)

  // Memoize the image source conversion
  const imageSrc = useMemo(() => getImageSrc(image.url), [image.url])

  const handleCopyPrompt = async () => {
    await navigator.clipboard.writeText(image.prompt)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleDownload = () => {
    const link = document.createElement('a')
    link.href = imageSrc
    link.download = `generated-image-${image.seed || Date.now()}.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <Card className="overflow-hidden transition-shadow hover:shadow-md">
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-2">
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <ImageIcon className="h-4 w-4" />
            Generated Image
          </CardTitle>
          <Badge variant="outline" className="text-xs">
            {image.resolution}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 p-0">
        <div className="relative aspect-square w-full bg-muted">
          {!imageError && imageSrc ? (
            <img
              src={imageSrc}
              alt={image.prompt}
              className="h-full w-full object-cover"
              onError={() => setImageError(true)}
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center text-muted-foreground">
              <ImageIcon className="h-12 w-12" />
            </div>
          )}
        </div>

        <div className="space-y-3 p-4 pt-0">
          <p className="line-clamp-3 text-sm text-muted-foreground">
            {image.prompt}
          </p>

          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>Seed: {image.seed}</span>
          </div>

          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={handleDownload}>
              <Download className="mr-1 h-3 w-3" />
              Download
            </Button>
            <Button variant="outline" size="sm" onClick={handleCopyPrompt}>
              {copied ? (
                <>
                  <Check className="mr-1 h-3 w-3" />
                  Copied
                </>
              ) : (
                <>
                  <Copy className="mr-1 h-3 w-3" />
                  Copy Prompt
                </>
              )}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export default function ImageGallery({ images }: ImageGalleryProps) {
  if (!images?.length) {
    return null
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold flex items-center gap-2">
        <ImageIcon className="h-5 w-5" />
        Generated Images ({images.length})
      </h3>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {images.map((image, index) => (
          <ImageCard key={`${image.seed}-${index}`} image={image} />
        ))}
      </div>
    </div>
  )
}
