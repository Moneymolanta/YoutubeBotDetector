"use client"

import React, { useState, useEffect } from "react"
import { VideoPlayer } from "@/components/video-player"
import { CommentsList } from "@/components/comments-list"
import { BotAnalyticsGraph } from "@/components/bot-analytics-graph"
import { Loader2, Bot, MessageSquare } from "lucide-react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { decodeHtml, decodeHtmlToString } from "@/lib/html-utils"

interface VideoPageProps {
  params: Promise<{
    videoId: string
  }>
}

interface VideoDetails {
  title: string
  channelTitle: string
  viewCount: string
  likeCount: string
  publishedAt: string
}

export default function VideoPage({ params }: VideoPageProps) {
  // Next.js 15+: params is a Promise, must unwrap with React.use
  const { videoId } = React.use(params);
  const [videoDetails, setVideoDetails] = useState<VideoDetails | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [formattedDate, setFormattedDate] = useState<string>("");

  useEffect(() => {
    setError(null); // Clear error when videoId changes or component mounts
    const fetchVideoDetails = async () => {
      setLoading(true)
      setError(null)

      try {
        const FLASK_BASE_URL = process.env.NEXT_PUBLIC_FLASK_BASE_URL!;
        const response = await fetch(`${FLASK_BASE_URL}/api/video?videoId=${videoId}`);

        if (!response.ok) {
          throw new Error("Failed to fetch video details");
        }

        const data = await response.json();
        setVideoDetails(data.videoDetails);
      } catch (err) {
        setError("Error loading video details. Please try again.")
        console.error(err)
      } finally {
        setLoading(false)
      }
    }

    fetchVideoDetails()
  }, [videoId])

  useEffect(() => {
    if (videoDetails?.publishedAt) {
      setFormattedDate(new Date(videoDetails.publishedAt).toLocaleDateString());
    } else {
      setFormattedDate("");
    }
  }, [videoDetails?.publishedAt]);

  return (
    <main className="min-h-screen p-4 md:p-8 bg-gray-50">
      <div className="max-w-7xl mx-auto space-y-8">
        {loading && !videoDetails ? (
          <div className="flex justify-center items-center py-10">
            <Loader2 className="h-8 w-8 animate-spin text-primary mr-2" />
            <span>Loading video details...</span>
          </div>
        ) : error ? (
          <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-md">{error}</div>
        ) : (
          <div className="space-y-8">
            <div className="bg-white rounded-lg shadow-sm overflow-hidden">
              <VideoPlayer videoId={videoId} title={decodeHtmlToString(videoDetails?.title) || ""} />

              <div className="p-4">
                <h1 className="text-xl font-bold">{decodeHtml(videoDetails?.title)}</h1>

                <div className="flex flex-wrap items-center gap-4 mt-3">
                  <Badge variant="outline" className="flex items-center gap-1">
                    <span>{decodeHtml(videoDetails?.channelTitle)}</span>
                  </Badge>

                  <div className="flex items-center gap-1 text-sm text-gray-500">
                    <span>👁️ {Number.parseInt(videoDetails?.viewCount || "0").toLocaleString()} views</span>
                  </div>

                  <div className="flex items-center gap-1 text-sm text-gray-500">
                    <span>👍 {Number.parseInt(videoDetails?.likeCount || "0").toLocaleString()}</span>
                  </div>

                  <div className="flex items-center gap-1 text-sm text-gray-500">
                    <span>📅 {formattedDate}</span>
                  </div>
                </div>
              </div>
            </div>

            <Tabs defaultValue="analytics" className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="analytics">
                  <Bot className="h-4 w-4 mr-2" /> Bot Analytics
                </TabsTrigger>
                <TabsTrigger value="comments">
                  <MessageSquare className="h-4 w-4 mr-2" /> Comments
                </TabsTrigger>
              </TabsList>
              <TabsContent value="analytics" className="py-4">
                <div className="space-y-6">
                  <BotAnalyticsGraph videoId={videoId} />
                </div>
              </TabsContent>
              <TabsContent value="comments" className="py-4 px-4 bg-white rounded-lg shadow-sm">
                <CommentsList videoId={videoId} />
              </TabsContent>
            </Tabs>
          </div>
        )}
      </div>
    </main>
  )
}
