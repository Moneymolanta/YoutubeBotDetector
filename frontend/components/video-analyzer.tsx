"use client";
import React, { useState } from "react";

export default function VideoAnalyzer() {
  const [url, setUrl] = useState("");
  const [model, setModel] = useState("bert");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const FLASK_BASE_URL = process.env.NEXT_PUBLIC_FLASK_BASE_URL!;
      const res = await fetch(`${FLASK_BASE_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, model }),
      });
      const data = await res.json();
      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
      }
    } catch (err) {
      setError("Request failed. Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto p-4">
      <form onSubmit={handleAnalyze} className="flex flex-col gap-4">
        <input
          type="text"
          placeholder="Enter YouTube video URL"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          className="border p-2 rounded"
          required
        />
        <select value={model} onChange={(e) => setModel(e.target.value)} className="border p-2 rounded">
          <option value="bert">BERT</option>
          <option value="cnn">CNN</option>
          <option value="rnn">RNN</option>
        </select>
        <button type="submit" className="bg-blue-500 text-white rounded p-2" disabled={loading}>
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </form>
      {error && <div className="text-red-500 mt-4">{error}</div>}
      {result && (
        <div className="mt-6 bg-white rounded shadow p-4">
          <h2 className="text-xl font-bold mb-2">{result.video_title}</h2>
          <p>
            <strong>Total Comments:</strong> {result.total}
          </p>
          <p>
            <strong>Bot Comments:</strong> {result.bots}
          </p>
          <p>
            <strong>Non-Bot Comments:</strong> {result.non_bots}
          </p>
          {result.label_plot && (
            <>
              <h3 className="mt-4 font-semibold">Prediction Distribution</h3>
              <img src={`data:image/png;base64,${result.label_plot}`} alt="Prediction Distribution" />
            </>
          )}
          {result.conf_matrix_plot && (
            <>
              <h3 className="mt-4 font-semibold">Confusion Matrix</h3>
              <img src={`data:image/png;base64,${result.conf_matrix_plot}`} alt="Confusion Matrix" />
            </>
          )}
        </div>
      )}
    </div>
  );
}
