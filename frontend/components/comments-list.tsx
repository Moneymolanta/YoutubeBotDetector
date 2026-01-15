   "use client"

import { useState, useEffect } from "react"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Card } from "@/components/ui/card"
import { Loader2, Bot, User, Filter } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { decodeHtml, decodeHtmlToString } from "@/lib/html-utils"

interface ModelResult {
  isBot: boolean
  confidence: number
  reason?: string
}

interface Comment {
  id: string
  snippet: {
    topLevelComment: {
      snippet: {
        authorDisplayName: string
        authorProfileImageUrl: string
        textDisplay: string
        likeCount: number
        publishedAt: string
      }
    }
  }
  modelResults: {
    cnnModel: ModelResult
    rnnModel: ModelResult
    bertModel: ModelResult
  }
}

interface CommentsListProps {
  videoId: string
}

export function CommentsList({ videoId }: CommentsListProps) {
  const [comments, setComments] = useState<Comment[]>([])
  const [filteredComments, setFilteredComments] = useState<Comment[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [stats, setStats] = useState({ total: 0, bots: 0, non_bots: 0 })
  const [filter, setFilter] = useState("all")

  useEffect(() => {
    const fetchComments = async () => {
  console.debug('[CommentsList] Fetching comments for videoId:', videoId);
      if (!videoId) return

      setLoading(true)
      setError(null)

      try {
        const FLASK_BASE_URL = process.env.NEXT_PUBLIC_FLASK_BASE_URL!;
        console.debug('[CommentsList] Sending request to:', `${FLASK_BASE_URL}/analyze`);
        const response = await fetch(`${FLASK_BASE_URL}/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: `https://www.youtube.com/watch?v=${videoId}`, model: 'all' }),
        })

        console.debug('[CommentsList] Received response status:', response.status);
        if (!response.ok) {
          throw new Error('Failed to fetch comments')
        }

        const data = await response.json();
        if (data) {
          console.debug('[CommentsList] Data retrieved from backend:', data);
        } else {
          console.warn('[CommentsList] No data received from backend.');
        }
        // Check for backend error and surface it
        if (data.error) {
          setError(data.error)
          setComments([])
          setLoading(false)
          return
        }

        // Parse enriched comments from the /analyze response
        let commentsArr: Comment[] = []
        if (Array.isArray(data.comments)) {
          commentsArr = data.comments
        } else if (Array.isArray(data)) {
          commentsArr = data
        }
        
        // Process comments to ensure they have the expected structure
        commentsArr = commentsArr.map(comment => {
          // If modelResults doesn't exist, create default ones
          if (!comment.modelResults) {
            comment.modelResults = {
              cnnModel: { isBot: false, confidence: 0 },
              rnnModel: { isBot: false, confidence: 0 },
              bertModel: { isBot: false, confidence: 0 }
            }
          }
          return comment
        })
        setComments(commentsArr)
        setFilteredComments(commentsArr)

        // Calculate stats - a comment is considered a bot if at least 2 models say so
        const total = commentsArr.length
        const bots = commentsArr.filter((c: Comment) => {
          const botVotes = [
            c.modelResults?.cnnModel?.isBot ?? false,
            c.modelResults?.rnnModel?.isBot ?? false,
            c.modelResults?.bertModel?.isBot ?? false,
          ].filter(Boolean).length
          return botVotes >= 2
        }).length

        setStats({
          total,
          bots,
          non_bots: total - bots,
        })
      } catch (err) {
        console.error('[CommentsList] Fetch error:', err);
        setError("Error loading comments. Please try again.")
      } finally {
        setLoading(false)
      }
    }

    console.debug('[CommentsList] Starting fetchComments()');
    fetchComments()
    console.debug('[CommentsList] Finished fetchComments()');
  }, [videoId])

  useEffect(() => {
    if (filter === "all") {
      setFilteredComments(comments)
    } else if (filter === "bots") {
      setFilteredComments(
        comments.filter((comment) => {
          const botVotes = [
            comment.modelResults?.cnnModel?.isBot ?? false,
            comment.modelResults?.rnnModel?.isBot ?? false,
            comment.modelResults?.bertModel?.isBot ?? false,
          ].filter(Boolean).length
          return botVotes >= 2
        })
      )
    } else if (filter === "humans") {
      setFilteredComments(
        comments.filter((comment) => {
          const botVotes = [
            comment.modelResults?.cnnModel?.isBot ?? false,
            comment.modelResults?.rnnModel?.isBot ?? false,
            comment.modelResults?.bertModel?.isBot ?? false,
          ].filter(Boolean).length
          return botVotes < 2
        })
      )
    } else if (filter === "cnn") {
      setFilteredComments(comments.filter((comment) => comment.modelResults?.cnnModel?.isBot ?? false))
    } else if (filter === "rnn") {
      setFilteredComments(comments.filter((comment) => comment.modelResults?.rnnModel?.isBot ?? false))
    } else if (filter === "bert") {
      setFilteredComments(comments.filter((comment) => comment.modelResults?.bertModel?.isBot ?? false))
    }
  }, [filter, comments])

  if (loading) {
    return (
      <div className="flex justify-center items-center py-10">
        <Loader2 className="h-8 w-8 animate-spin text-primary mr-2" />
        <span>Loading comments...</span>
      </div>
    )
  }

  if (error) {
    return <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-md">{error}</div>
  }

  if (comments.length === 0 && !loading) {
    return <div className="text-center py-10 text-gray-500">No comments found for this video.</div>
  }

  const getBotScore = (comment: Comment) => {
    const botVotes = [
      comment.modelResults?.cnnModel?.isBot ?? false,
      comment.modelResults?.rnnModel?.isBot ?? false,
      comment.modelResults?.bertModel?.isBot ?? false,
    ].filter(Boolean).length
    return botVotes
  }

  return (
    <div className="space-y-6">
      <div className="bg-white p-4 rounded-lg shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-xl font-semibold">Comments ({stats.total})</h2>
            <div className="flex gap-4 mt-2">
              <div className="flex items-center">
                <div className="h-3 w-3 rounded-full bg-green-500 mr-2"></div>
                <span className="text-sm">Bots: {stats.bots}</span>
              </div>
              <div className="flex items-center">
                <span className="text-xs ml-4">Model legend: <span className="font-bold text-green-700">Human</span> / <span className="font-bold text-red-700">Bot</span></span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="divide-y">
        {filteredComments.map((comment) => {
          let author = "Unknown"
          let avatar: string | undefined = undefined
          let text = ""
          if (comment.snippet?.topLevelComment?.snippet) {
            author = comment.snippet.topLevelComment.snippet.authorDisplayName ?? "Unknown"
            avatar = comment.snippet.topLevelComment.snippet.authorProfileImageUrl
            text = comment.snippet.topLevelComment.snippet.textDisplay ?? ""
          } else if (comment.snippet) {
            author = (comment.snippet as any).authorDisplayName ?? "Unknown"
            avatar = (comment.snippet as any).authorProfileImageUrl
            text = (comment.snippet as any).textDisplay ?? ""
          } else if ((comment as any).authorDisplayName || (comment as any).text) {
            author = (comment as any).authorDisplayName ?? "Unknown"
            avatar = (comment as any).authorProfileImageUrl
            text = (comment as any).text ?? ""
          }
          return (
            <div key={comment.id} className="py-4">
              <div className="flex items-center gap-2">
                <Avatar>
                  <AvatarImage src={avatar} />
                  <AvatarFallback>{author[0]}</AvatarFallback>
                </Avatar>
                <span className="font-semibold">{author}</span>
              </div>
              <div className="mt-2">{decodeHtml(text)}</div>
              <div className="mt-2">
                <span className="text-xs text-gray-500">Model predictions:</span>
                <div className="flex flex-row gap-2 mt-1">
                  {comment.modelResults?.cnnModel && (
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Badge
                            variant={comment.modelResults.cnnModel.isBot ? "destructive" : "default"}
                            className={`text-xs flex items-center gap-1 ${comment.modelResults.cnnModel.isBot ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"}`}
                          >
                            {comment.modelResults.cnnModel.isBot ? <Bot className="h-3 w-3" /> : <User className="h-3 w-3" />} CNN
                          </Badge>
                        </TooltipTrigger>
                        <TooltipContent>
                          {comment.modelResults.cnnModel.isBot ? (
                            <p>
                              <strong>Bot</strong> ({comment.modelResults.cnnModel.confidence.toFixed(1)}%)<br />
                              Reason: {comment.modelResults.cnnModel.reason}
                            </p>
                          ) : (
                            <p>
                              <strong>Human</strong> ({comment.modelResults.cnnModel.confidence.toFixed(1)}%)<br />
                              Reason: {comment.modelResults.cnnModel.reason}
                            </p>
                          )}
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  )}
                  {comment.modelResults?.rnnModel && (
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Badge
                            variant={comment.modelResults.rnnModel.isBot ? "destructive" : "default"}
                            className={`text-xs flex items-center gap-1 ${comment.modelResults.rnnModel.isBot ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"}`}
                          >
                            {comment.modelResults.rnnModel.isBot ? <Bot className="h-3 w-3" /> : <User className="h-3 w-3" />} RNN
                          </Badge>
                        </TooltipTrigger>
                        <TooltipContent>
                          {comment.modelResults.rnnModel.isBot ? (
                            <p>
                              <strong>Bot</strong> ({comment.modelResults.rnnModel.confidence.toFixed(1)}%)<br />
                              Reason: {comment.modelResults.rnnModel.reason}
                            </p>
                          ) : (
                            <p>
                              <strong>Human</strong> ({comment.modelResults.rnnModel.confidence.toFixed(1)}%)<br />
                              Reason: {comment.modelResults.rnnModel.reason}
                            </p>
                          )}
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  )}
                  {comment.modelResults?.bertModel && (
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Badge
                            variant={comment.modelResults.bertModel.isBot ? "destructive" : "default"}
                            className={`text-xs flex items-center gap-1 ${comment.modelResults.bertModel.isBot ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"}`}
                          >
                            {comment.modelResults.bertModel.isBot ? <Bot className="h-3 w-3" /> : <User className="h-3 w-3" />} BERT
                          </Badge>
                        </TooltipTrigger>
                        <TooltipContent>
                          {comment.modelResults.bertModel.isBot ? (
                            <p>
                              <strong>Bot</strong> ({comment.modelResults.bertModel.confidence.toFixed(1)}%)<br />
                              Reason: {comment.modelResults.bertModel.reason}
                            </p>
                          ) : (
                            <p>
                              <strong>Human</strong> ({comment.modelResults.bertModel.confidence.toFixed(1)}%)<br />
                              Reason: {comment.modelResults.bertModel.reason}
                            </p>
                          )}
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  )}
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}