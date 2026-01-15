"use client"

import type React from "react"

import Link from "next/link"
import { usePathname, useRouter, useSearchParams } from "next/navigation"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Search, Bot, BarChart2, Home } from "lucide-react"

export function Navbar() {
  const router = useRouter()
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const [query, setQuery] = useState(searchParams.get("q") || "")

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return
    router.push(`/?q=${encodeURIComponent(query)}`)
  }

  return (
    <nav className="bg-white border-b shadow-sm sticky top-0 z-10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link href="/" className="flex-shrink-0 flex items-center">
              <Bot className="h-6 w-6 text-primary mr-2" />
              <span className="font-semibold text-xl">YouTube Bot Detector</span>
            </Link>
          </div>
          <div className="flex items-center gap-4">
            <form onSubmit={handleSearch} className="flex gap-2 relative">
              <Input
                type="text"
                placeholder="Search videos..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="w-64"
              />
              <Button type="submit" size="sm" variant="ghost">
                <Search className="h-4 w-4" />
              </Button>
            </form>
            <div className="flex gap-2">
              <Link href="/">
                <Button variant="ghost" className={`${pathname === "/" ? "bg-accent" : ""}`}>
                  <Home className="h-4 w-4 mr-1" /> Home
                </Button>
              </Link>
              <Link href="/about">
                <Button variant="ghost" className={`${pathname === "/about" ? "bg-accent" : ""}`}>
                  <BarChart2 className="h-4 w-4 mr-1" /> About
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </div>
    </nav>
  )
}
