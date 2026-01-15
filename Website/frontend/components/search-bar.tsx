"use client"

import type React from "react"

import { useState } from "react"
import { Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
import Image from "next/image"

interface SearchResult {
  id: {
    videoId: string
  }
  snippet: {
    title: string
    description: string
    thumbnails: {
      medium: {
        url: string
      }
    }
    channelTitle: string
  }
}

interface SearchBarProps {
  onVideoSelect: (videoId: string, title: string) => void
}

export function SearchBar({ onVideoSelect }: SearchBarProps) {
  const [query, setQuery] = useState("")
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setError(null)

    try {
      const FLASK_BASE_URL = process.env.NEXT_PUBLIC_FLASK_API_URL || 'http://localhost:5001';
      const response = await fetch(`${FLASK_BASE_URL}/api/search?q=${encodeURIComponent(query)}`)

      if (!response.ok) {
        throw new Error("Failed to search videos")
      }

      const data = await response.json()
      setResults(data.items || [])
    } catch (err) {
      setError("Error searching videos. Please try again.")
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <form onSubmit={handleSearch} className="flex gap-2">
        <Input
          type="text"
          placeholder="Search for YouTube videos..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="flex-1"
        />
        <Button type="submit" disabled={loading}>
          {loading ? "Searching..." : "Search"}
          <Search className="ml-2 h-4 w-4" />
        </Button>
      </form>

      {error && <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-md">{error}</div>}

      {results.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {results.map((video) => (
            <Card
              key={video.id.videoId}
              className="overflow-hidden cursor-pointer hover:shadow-md transition-shadow"
              onClick={() => onVideoSelect(video.id.videoId, video.snippet.title)}
            >
              <div className="aspect-video relative">
                <Image
                  src={video.snippet.thumbnails.medium.url || "/placeholder.svg"}
                  alt={video.snippet.title}
                  fill
                  className="object-cover"
                />
              </div>
              <div className="p-4">
                <h3 className="font-semibold line-clamp-2">{video.snippet.title}</h3>
                <p className="text-sm text-gray-500 mt-1">{video.snippet.channelTitle}</p>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
