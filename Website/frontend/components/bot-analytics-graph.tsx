"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Loader2 } from "lucide-react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  Cell,
  PieChart as RechartsPieChart,
  Pie,
} from "recharts"
import { ChartContainer } from "@/components/ui/chart"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import VideoAnalyzer from "@/components/video-analyzer"

interface BotAnalyticsGraphProps {
  videoId: string
}

interface AnalyticsData {
  name: string
  bots: number
  non_bots: number
}

interface AgreementData {
  name: string
  value: number
  color: string
}

export function BotAnalyticsGraph({ videoId }: BotAnalyticsGraphProps) {
  const [model, setModel] = useState<string>("bert");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [videoTitle, setVideoTitle] = useState<string>("");
  const [total, setTotal] = useState<number>(0);
  const [bots, setBots] = useState<number>(0);
  const [nonBots, setNonBots] = useState<number>(0);
  const [labelPlot, setLabelPlot] = useState<string | null>(null);
  const [confMatrixPlot, setConfMatrixPlot] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnalytics = async () => {
      setLoading(true);
      setError(null);
      try {
        const FLASK_BASE_URL = process.env.NEXT_PUBLIC_FLASK_BASE_URL!;
        const videoUrl = `https://www.youtube.com/watch?v=${videoId}`;
        const response = await fetch(`${FLASK_BASE_URL}/analyze`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: videoUrl, model }),
        });
        if (!response.ok) {
          throw new Error("Failed to fetch analytics data");
        }
        const data = await response.json();
        console.log(data)
        setVideoTitle(data.video_title || "");
        setTotal(data.total || 0);
        setBots(data.bots || 0);
        setNonBots(data.non_bots || 0);
        setLabelPlot(data.label_plot || null);
        setConfMatrixPlot(data.conf_matrix_plot || null);
      } catch (err) {
        setError("Error loading analytics data. Please try again.");
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchAnalytics();
  }, [videoId, model]);

  if (loading) {
    return (
      <div className="flex justify-center items-center py-10">
        <Loader2 className="h-8 w-8 animate-spin text-primary mr-2" />
        <span>Loading analytics data...</span>
      </div>
    )
  }

  if (error) {
    return <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-md">{error}</div>
  }

  return (
    <div className="grid gap-6">
          <Card>
            <CardContent>
              <div className="mb-6">
                <label htmlFor="model-select" className="block text-sm font-medium mb-2 mt-4">Select Model</label>
                <select
                  id="model-select"
                  value={model}
                  onChange={e => setModel(e.target.value)}
                  className="border p-2 rounded"
                >
                  <option value="bert">BERT</option>
                  <option value="cnn">CNN</option>
                  <option value="rnn">RNN</option>
                </select>
              </div>
              {videoTitle && (
                <div className="mb-4">
                  {/* <h2 className="text-lg font-bold">{videoTitle}</h2> */}
                  <div className="flex flex-wrap gap-4 mt-2">
                    <span className="bg-gray-100 rounded px-3 py-1 text-sm">Total Comments: {total}</span>
                    <span className="bg-green-100 rounded px-3 py-1 text-sm text-green-800">Human Comments: {nonBots}</span>
                    <span className="bg-red-100 rounded px-3 py-1 text-sm text-red-800">Bot Comments: {bots}</span>
                  </div>
                </div>
              )}
              {labelPlot && (
                <div className="my-4">
                  {/* <h3 className="font-semibold mb-2">Prediction Distribution</h3> */}
                  <img src={`data:image/png;base64,${labelPlot}`} alt="Prediction Distribution" className="rounded border" />
                </div>
              )}
              {confMatrixPlot && (
                <div className="my-4">
                  <h3 className="font-semibold mb-2">Confusion Matrix</h3>
                  <img src={`data:image/png;base64,${confMatrixPlot}`} alt="Confusion Matrix" className="rounded border" />
                </div>
              )}
            </CardContent>
          </Card>
    </div>
  )
}
