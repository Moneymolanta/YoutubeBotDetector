"use client"

import type React from "react"
import { useState } from "react"
import { Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { useRouter } from "next/navigation"

interface SharedSearchBarProps {
  defaultQuery?: string
}

export function SharedSearchBar({ defaultQuery = "" }: SharedSearchBarProps) {
  const [query, setQuery] = useState(defaultQuery)
  const [loading, setLoading] = useState(false)
  const router = useRouter()

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    router.push(`/?q=${encodeURIComponent(query)}`)
    setLoading(false)
  }

  return (
    <form onSubmit={handleSearch} className="flex gap-2 w-full max-w-2xl mx-auto">
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
  )
}
