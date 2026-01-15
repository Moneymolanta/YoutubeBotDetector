"use client"

import { decodeHtmlToString } from "@/lib/html-utils"

interface VideoPlayerProps {
  videoId: string
  title: string
}

export function VideoPlayer({ videoId, title }: VideoPlayerProps) {
  return (
    <div className="aspect-video w-full bg-black">
      <iframe
        src={`https://www.youtube.com/embed/${videoId}`}
        title={decodeHtmlToString(title)}
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowFullScreen
        className="w-full h-full"
      ></iframe>
    </div>
  )
}
