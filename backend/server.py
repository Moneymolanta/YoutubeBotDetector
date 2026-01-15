from flask import Flask, request, jsonify, send_file
from Trained_bert import run_inference_on_video as run_bert

# Trained_cnn now handles model/vocab caching and batch inference efficiently
from Trained_cnn import run_cnn_inference_on_video
from Trained_rnn import run_rnn_inference_on_video
import re
from flask_cors import CORS
import requests
import os
import time

CACHE_TTL = 3600  # cache results for 1 hour (in seconds)
video_analysis_cache = {}

# --- Recent Searches (in-memory, not persistent) ---
RECENT_SEARCHES_MAX = 10
recent_searches = []


def get_cached_result(video_id, model_type):
    cache_key = f"{video_id}_{model_type}"
    entry = video_analysis_cache.get(cache_key)
    if entry:
        timestamp, result = entry
        if time.time() - timestamp < CACHE_TTL:
            return result
        else:
            # Expired, remove from cache
            del video_analysis_cache[cache_key]
    return None


def cache_result(video_id, model_type, result):
    cache_key = f"{video_id}_{model_type}"
    video_analysis_cache[cache_key] = (time.time(), result)


app = Flask(__name__)
app.url_map.strict_slashes = False

# Enable CORS for all routes with proper configuration
CORS(
    app,
    origins="*",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Origin", "Accept"],
    supports_credentials=False,
    send_wildcard=True
)


YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")

if not YOUTUBE_API_KEY:
    raise RuntimeError(
        "YOUTUBE_API_KEY environment variable not set. Please set it to your YouTube API key."
    )

# --- API Endpoints for Next.js frontend ---
@app.route("/api/analytics")
def analytics():
    video_id = request.args.get("videoId")
    if not video_id:
        return jsonify({"error": "Video ID is required"}), 400
    # Fetch comments from YouTube API
    comments_url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&maxResults=100&key={YOUTUBE_API_KEY}"
    resp = requests.get(comments_url)
    if resp.status_code != 200:
        return jsonify({"error": "Failed to fetch comments"}), resp.status_code
    comments = resp.json()
    # TODO: Add analytics/model logic here if needed
    return jsonify(comments)


@app.route("/api/comments")
def comments():
    video_id = request.args.get("videoId")
    if not video_id:
        return jsonify({"error": "Video ID is required"}), 400
    comments_url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&maxResults=100&key={YOUTUBE_API_KEY}"
    resp = requests.get(comments_url)
    if resp.status_code != 200:
        return jsonify({"error": "Failed to fetch comments"}), resp.status_code
    return jsonify(resp.json())


@app.route("/api/search")
def search():
    query = request.args.get("q")
    if not query:
        return jsonify({"error": "Search query is required"}), 400
    search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=9&q={query}&type=video&key={YOUTUBE_API_KEY}"
    resp = requests.get(search_url)
    print("YouTube API response:", resp.status_code, resp.text)
    if resp.status_code != 200:
        return jsonify({"error": "Failed to fetch videos"}), resp.status_code
    # Log recent search (avoid duplicates in a row, trim to max length)
    global recent_searches
    if query:
        if not recent_searches or recent_searches[0] != query:
            recent_searches.insert(0, query)
            if len(recent_searches) > RECENT_SEARCHES_MAX:
                del recent_searches[RECENT_SEARCHES_MAX:]
    return jsonify(resp.json())


@app.route("/api/recent-searches")
def get_recent_searches():
    """Return list of recent search queries (most recent first)"""
    return jsonify({"recent": recent_searches})


@app.route("/api/video")
def video():
    video_id = request.args.get("videoId")
    if not video_id:
        return jsonify({"error": "Video ID is required"}), 400
    video_url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={video_id}&key={YOUTUBE_API_KEY}"
    resp = requests.get(video_url)
    print("YouTube API response:", resp.status_code, resp.text)
    if resp.status_code != 200:
        return jsonify({"error": "Failed to fetch video details"}), resp.status_code
    data = resp.json()
    if not data.get("items"):
        return jsonify({"error": "Video not found"}), 404
    item = data["items"][0]
    video_details = {
        "title": item["snippet"]["title"],
        "channelTitle": item["snippet"]["channelTitle"],
        "publishedAt": item["snippet"]["publishedAt"],
        "viewCount": item["statistics"]["viewCount"],
        "likeCount": item["statistics"].get("likeCount"),
    }
    return jsonify({"videoDetails": video_details})


def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None


@app.route("/")
def index():
    # Return API info instead of trying to serve a frontend file
    return jsonify({
        "api": "YouTube Bot Detector API",
        "version": "1.0",
        "endpoints": [
            "/api/search",
            "/api/video",
            "/api/comments",
            "/api/analytics",
            "/api/recent-searches",
            "/analyze"
        ]
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    video_url = data.get("url")
    model_type = data.get("model", "bert")

    video_id = extract_video_id(video_url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    # Check cache first
    cached = get_cached_result(video_id, model_type)
    if cached:
        return jsonify(cached)

    # If 'all', run all models and aggregate per-comment predictions
    if model_type == "all":
        # Only aggregate RNN and BERT results (skip CNN)
        rnn_result = run_rnn_inference_on_video(video_id)
        bert_result = run_bert(video_id)
        cnn_result = run_cnn_inference_on_video(video_id)

        comments = []
        try:
            rnn_comments = rnn_result.get("comments", [])
            bert_comments = bert_result.get("comments", [])
            cnn_comments = cnn_result.get("comments", [])

            # Use RNN as base, fallback to BERT if empty
            base_comments = rnn_comments if rnn_comments else bert_comments
            for i, base_comment in enumerate(base_comments):
                comment_obj = {
                    "id": base_comment.get("id", str(i)),
                    "text": base_comment.get("text", ""),
                    "modelResults": {
                        "rnnModel": (
                            rnn_comments[i]["modelResults"]["rnnModel"]
                            if i < len(rnn_comments)
                            and "modelResults" in rnn_comments[i]
                            else {"isBot": False, "confidence": 0, "reason": ""}
                        ),
                        "bertModel": (
                            bert_comments[i]["modelResults"]["bertModel"]
                            if i < len(bert_comments)
                            and "modelResults" in bert_comments[i]
                            else {"isBot": False, "confidence": 0, "reason": ""}
                        ),
                        "cnnModel": (
                            cnn_comments[i]["modelResults"]["cnnModel"]
                            if i < len(cnn_comments)
                            and "modelResults" in cnn_comments[i]
                            else {"isBot": False, "confidence": 0, "reason": ""}
                        ),
                    },
                }
                comment_obj["snippet"] = base_comment.get("snippet", None)
                comments.append(comment_obj)
        except Exception as e:
            print(f"Error aggregating model results: {e}")
            comments = []
        # Compose summary from both models
        total = len(comments)
        bots = sum(
            [
                1
                for c in comments
                if sum(
                    [
                        c["modelResults"]["rnnModel"]["isBot"],
                        c["modelResults"]["bertModel"]["isBot"],
                        c["modelResults"]["cnnModel"]["isBot"],
                    ]
                )
                >= 1  # Mark as bot if either model says so
            ]
        )
        humans = total - bots
        result = {
            "comments": comments,
            "total": total,
            "bots": bots,
            "non_bots": humans,
            "label_plot": bert_result.get("label_plot"),
            "conf_matrix_plot": bert_result.get("conf_matrix_plot"),
            "video_title": bert_result.get("video_title", ""),
        }
    else:
        model_func = {
            "rnn": run_rnn_inference_on_video,
            "bert": run_bert,
            "cnn": run_cnn_inference_on_video,
        }.get(model_type, run_bert)
        model_result = model_func(video_id)
        comments = model_result.get("comments", [])
        # Compose stats
        total = len(comments)
        bots = sum([c["modelResults"][f"{model_type}Model"]["isBot"] for c in comments])
        humans = total - bots
        result = {
            "comments": comments,
            "total": total,
            "bots": bots,
            "non_bots": humans,
            "label_plot": model_result.get("label_plot"),
            "conf_matrix_plot": model_result.get("conf_matrix_plot"),
            "video_title": model_result.get("video_title", ""),
        }

    if isinstance(result, dict) and result.get("error"):
        return jsonify(result), 400
    cache_result(video_id, model_type, result)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
