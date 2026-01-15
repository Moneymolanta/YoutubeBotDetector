"use client"

import { useState, useEffect } from "react"
import { Card } from "@/components/ui/card"
import { Loader2, Bot } from "lucide-react"
import Image from "next/image"
import Link from "next/link"
import { useSearchParams } from "next/navigation"
import { decodeHtml, decodeHtmlToString } from "@/lib/html-utils"

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

export default function Home() {
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [recent, setRecent] = useState<string[]>([])
  const searchParams = useSearchParams()
  const query = searchParams.get("q") || ""
  const router = typeof window !== 'undefined' ? require('next/navigation').useRouter() : null;

  useEffect(() => {
    // Fetch recent searches on mount
    const fetchRecent = async () => {
      try {
        const FLASK_BASE_URL = process.env.NEXT_PUBLIC_FLASK_BASE_URL!;
        const res = await fetch(`${FLASK_BASE_URL}/api/recent-searches`)
        if (res.ok) {
          const data = await res.json()
          setRecent(data.recent || [])
        }
      } catch (e) {
        // fail silently
      }
    }
    fetchRecent()
  }, [])

  useEffect(() => {
    // If query is empty, clear results so recent searches show
    if (!query) {
      setResults([])
      return
    }
    const fetchResults = async () => {
  console.debug('[Home] Starting fetchResults for query:', query);
      if (!query) return

      setLoading(true)
      setError(null)

      try {
        const FLASK_BASE_URL = process.env.NEXT_PUBLIC_FLASK_BASE_URL!;
        console.debug('[Home] Sending search request to:', `${FLASK_BASE_URL}/api/search?q=${encodeURIComponent(query)}`);
    const response = await fetch(`${FLASK_BASE_URL}/api/search?q=${encodeURIComponent(query)}`)

        console.debug('[Home] Received response status:', response.status);
    if (!response.ok) {
          throw new Error("Failed to search videos")
        }

        const data = await response.json();
    if (data) {
      console.debug('[Home] Search data retrieved from backend:', data);
    } else {
      console.warn('[Home] No search data received from backend.');
    }
    setResults(data.items || [])
      } catch (err) {
        setError("Error searching videos. Please try again.")
    console.error('[Home] Search fetch error:', err)
      } finally {
        setLoading(false)
      }
    }

    if (query) {
      fetchResults()
    }
  }, [query])

  return (
    <main className="min-h-screen p-4 md:p-8 bg-gray-50">
      <div className="max-w-7xl mx-auto space-y-8">
        <div className="text-center space-y-2 py-6">
          {recent.length > 0 && results.length === 0 && !loading && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold mb-2">Recent Searches</h3>
              <div className="flex flex-wrap gap-2 justify-center">
                {recent.map((item, idx) => (
                  <button
                    key={item + idx}
                    className="px-3 py-1 rounded bg-gray-200 hover:bg-blue-100 text-sm text-blue-900 transition"
                    onClick={() => {
                      setResults([]); // Hide recent searches instantly
                      if (router) router.push(`/?q=${encodeURIComponent(item)}`)
                    }}
                    type="button"
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {!query && !loading && (
          <div className="flex flex-col items-center justify-center py-8">
            <Bot className="h-16 w-16 text-primary mb-4" />
            <div className="max-w-2xl text-center">
              <h2 className="text-xl font-semibold mb-3">Detect Bot Comments with AI</h2>
              <p className="text-gray-600 mb-6">
                Our platform uses three AI models (CNN, RNN, and BERT) to analyze YouTube comments and identify
                potential bot activity.
              </p>

            </div>
          </div>
        )}

        {loading && (
          <div className="flex justify-center items-center py-16">
            <Loader2 className="h-8 w-8 animate-spin text-primary mr-2" />
            <span>Searching for videos...</span>
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-md">
            <p>{error}</p>
          </div>
        )}

        {query && results.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {results.map((video) => (
              <Link href={`/video/${video.id.videoId}`} key={video.id.videoId}>
                <Card className="overflow-hidden cursor-pointer hover:shadow-md transition-all duration-300 card-hover h-full">
                  <div className="aspect-video relative">
                    <Image
                      src={video.snippet.thumbnails.medium.url || "/placeholder.svg"}
                      alt={video.snippet.title}
                      fill
                      className="object-cover"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 hover:opacity-100 transition-opacity duration-300 flex items-end">
                      <div className="p-4 text-white">
                        <p className="font-medium">Analyze comments</p>
                      </div>
                    </div>
                  </div>
                  <div className="p-4">
                    <h3 className="font-semibold line-clamp-2">{decodeHtml(video.snippet.title)}</h3>
                    <p className="text-sm text-gray-500 mt-1">{decodeHtml(video.snippet.channelTitle)}</p>
                  </div>
                </Card>
              </Link>
            ))}
          </div>
        )}

        {!loading && !error && query && results.length === 0 && (
          <div className="text-center py-10 text-gray-500">
            <p>No videos found for "{query}"</p>
          </div>
        )}
      </div>
    </main>
  )
}
