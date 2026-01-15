import os
import json
import re
import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from Model_bert import BERTWithMetadata
from datetime import datetime, timezone
import requests
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import base64
from io import BytesIO

# Load environment variables if needed.
load_dotenv()
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")  # or set directly: "YOUR_YOUTUBE_API_KEY"
if not YOUTUBE_API_KEY:
    print("WARNING: YOUTUBE_API_KEY is not set or is empty.")

# Helper for YouTube API requests
def youtube_api_request(endpoint, params):
    base_url = f"https://www.googleapis.com/youtube/v3/{endpoint}"
    params["key"] = YOUTUBE_API_KEY
    resp = requests.get(base_url, params=params)
    resp.raise_for_status()
    return resp.json()

# Load the saved model.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metadata_dim = 4  # e.g., features: like_count, published_at, viewer_rating, has_link
model = BERTWithMetadata(metadata_dim=metadata_dim)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Prepare the tokenizer.
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load the scaler (from training) using pickle.
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def process_comment(snippet, comments, valid_file, excluded_file):
    text = snippet.get("textDisplay", "")
    if not text.strip():
        excluded_file.write(json.dumps({"reason": "empty text", "data": snippet}, indent=4))
        excluded_file.write("\n\n--- NEW COMMENT ---\n\n")
        return

    like_count = snippet.get("likeCount", 0)
    published_str = snippet.get("publishedAt", "")
    try:
        try:
            published_dt = datetime.strptime(published_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        except ValueError:
            published_dt = datetime.strptime(published_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        published_numeric = published_dt.timestamp()
    except Exception as e:
        excluded_file.write(json.dumps({"reason": f"timestamp parse error: {e}", "data": snippet}, indent=4))
        excluded_file.write("\n\n--- NEW COMMENT ---\n\n")
        return

    viewer_rating = 0
    has_link = 1 if re.search(r'https?://\S+', text) else 0

    comment_dict = {
        "text": text,
        "like_count": like_count,
        "published_at": published_numeric,
        "viewer_rating": viewer_rating,
        "has_link": has_link
    }
    comments.append(comment_dict)
    valid_file.write(json.dumps(comment_dict, indent=4))
    valid_file.write("\n\n--- NEW COMMENT ---\n\n")


def fetch_api_comments(video_id, max_results=5000):
    valid_log_filename = f"commentsForAPI.txt"
    excluded_log_filename = f"excluded_commentsForAPI.txt"
    valid_file = open(valid_log_filename, "w", encoding="utf-8")
    excluded_file = open(excluded_log_filename, "w", encoding="utf-8")

    comments = []
    nextPageToken = None

    while len(comments) < max_results:
        current_max = min(100, max_results - len(comments))
        params = {
            "part": "snippet",
            "videoId": video_id,
            "textFormat": "plainText",
            "maxResults": current_max,
            "order": "relevance"
        }
        if nextPageToken:
            params["pageToken"] = nextPageToken
        try:
            response = youtube_api_request("commentThreads", params)
        except Exception as e:
            print(f"Error fetching comments: {e}")
            break

        for item in tqdm(response.get("items", []), desc="Fetching API Comments", leave=False):
            # Top-level comment
            top_snippet = item['snippet']['topLevelComment']['snippet']
            process_comment(top_snippet, comments, valid_file, excluded_file)

            # Replies (loop through all pages)
            if item['snippet'].get('totalReplyCount', 0) > 0:
                parent_id = item['snippet']['topLevelComment']['id']
                reply_next_page_token = None
                while True:
                    reply_params = {
                        "part": "snippet",
                        "parentId": parent_id,
                        "maxResults": 100
                    }
                    if reply_next_page_token:
                        reply_params["pageToken"] = reply_next_page_token
                    try:
                        reply_response = youtube_api_request("comments", reply_params)
                    except Exception as e:
                        print(f"Error fetching replies: {e}")
                        break
                    for reply in reply_response.get("items", []):
                        reply_snippet = reply['snippet']
                        process_comment(reply_snippet, comments, valid_file, excluded_file)
                    reply_next_page_token = reply_response.get("nextPageToken")
                    if not reply_next_page_token:
                        break
        nextPageToken = response.get("nextPageToken")
        if not nextPageToken:
            break
    valid_file.close()
    excluded_file.close()
    print(f"Valid comments logged to {valid_log_filename}")
    print(f"Excluded comments logged to {excluded_log_filename}")
    return pd.DataFrame(comments)


def save_plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64

# Specify the video ID for inference.
# --- Model, tokenizer, scaler loaded ONCE at module level ---
BERT_MODEL_PATH = "best_model.pth"
SCALER_PATH = "scaler.pkl"

# Load model once
_bert_model = BERTWithMetadata(metadata_dim=4)
_bert_model.load_state_dict(torch.load(BERT_MODEL_PATH, map_location=device))
_bert_model.to(device)
_bert_model.eval()

# Prepare the tokenizer once
_bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load the scaler once
with open(SCALER_PATH, "rb") as _f:
    _bert_scaler = pickle.load(_f)

# --- Comment-level cache (in-memory, per session) ---
_bert_comment_cache = {}


def run_inference_on_video(video_id, max_results=5000):
    # Get video title
    try:
        video_response = youtube_api_request("videos", {"part": "snippet", "id": video_id})
        items = video_response.get('items', [])
        video_title = items[0]['snippet']['title'] if items else "Unknown Title"
    except Exception as e:
        print(f"Error fetching video info: {e}")
        video_title = "Unknown Title"

    new_comments_df = fetch_api_comments(video_id, max_results)
    print("Fetched new comments:", new_comments_df.shape[0])

    meta_features = ['like_count', 'published_at', 'viewer_rating', 'has_link']
    if new_comments_df.empty or not all(col in new_comments_df.columns for col in meta_features):
        print("No comments fetched or missing expected columns. Returning error.")
        return {
            "error": "No comments fetched or missing expected columns. Check YouTube API key, quota, or video availability."
        }
    meta_data = new_comments_df[meta_features]
    meta_scaled = _bert_scaler.transform(meta_data)
    meta_tensor = torch.tensor(meta_scaled, dtype=torch.float)

    # --- Efficient Preprocessing (vectorized) and comment-level cache ---
    comment_results = []
    comments_to_process = []
    indices_to_process = []
    for idx, row in new_comments_df.iterrows():
        comment_text = row['text']
        cache_key = (video_id, comment_text)
        if cache_key in _bert_comment_cache:
            comment_results.append(_bert_comment_cache[cache_key])
        else:
            comment_results.append(None)
            comments_to_process.append(comment_text)
            indices_to_process.append(idx)

    # Only process uncached comments
    if comments_to_process:
        temp_df = new_comments_df.loc[indices_to_process].copy()
        temp_meta_data = temp_df[meta_features]
        temp_meta_scaled = _bert_scaler.transform(temp_meta_data)
        temp_meta_tensor = torch.tensor(temp_meta_scaled, dtype=torch.float)
        temp_encodings = _bert_tokenizer(list(temp_df['text']), padding=True, truncation=True, return_tensors='pt')

        class InferenceDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, metadata):
                self.encodings = encodings
                self.metadata = metadata
            def __getitem__(self, idx):
                return {
                    'input_ids': self.encodings['input_ids'][idx],
                    'attention_mask': self.encodings['attention_mask'][idx],
                    'metadata': self.metadata[idx]
                }
            def __len__(self):
                return len(self.metadata)

        temp_dataset = InferenceDataset(temp_encodings, temp_meta_tensor)
        temp_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=8, shuffle=False)
        predictions = []
        with torch.no_grad():
            for batch in tqdm(temp_loader, desc="Running inference"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                metadata = batch['metadata'].to(device)
                outputs = _bert_model(input_ids, attention_mask, metadata)
                preds = (outputs > 0.5).float()
                predictions.extend(preds.cpu().numpy())
        predictions = np.array(predictions).reshape(-1)
        # Store results in cache and update comment_results
        for idx2, pred in enumerate(predictions):
            cache_key = (video_id, comments_to_process[idx2])
            pred_label = int(pred)
            _bert_comment_cache[cache_key] = pred_label
            comment_results[indices_to_process[idx2]] = pred_label

    # Now, comment_results has all predictions in order
    new_comments_df['predicted_label'] = comment_results

    # Plot label distribution
    label_counts = new_comments_df['predicted_label'].value_counts().sort_index()
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(label_counts.index.astype(str), label_counts.values, color=['green', 'red'])
    ax1.set_xlabel("Predicted Label (0 = non-bot, 1 = bot)")
    ax1.set_ylabel("Number of Comments")
    ax1.set_title("Distribution of Predicted Labels")
    plt.tight_layout()
    label_dist_base64 = save_plot_to_base64(fig1)
    plt.close(fig1)

    # Confusion matrix
    conf_matrix_base64 = None
    if 'label' in new_comments_df.columns and new_comments_df['label'].nunique() > 1:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(new_comments_df['label'].values, new_comments_df['predicted_label'].values)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("True")
        ax2.set_title("Confusion Matrix")
        plt.tight_layout()
        conf_matrix_base64 = save_plot_to_base64(fig2)
        plt.close(fig2)

    # Build enriched comments array
    comments = []
    for idx, row in new_comments_df.iterrows():
        comment_obj = {
            'id': str(idx),
            'text': row.get('text', ''),
            'modelResults': {
                'bertModel': {
                    'isBot': bool(row['predicted_label']),
                    'confidence': 1.0,  # Placeholder, update if you have probability
                    'reason': ''  # Optionally add explanation if available
                }
            },
            'snippet': row.get('snippet', None)
        }
        comments.append(comment_obj)

    return {
        "video_title": video_title,
        "total": int(len(new_comments_df)),
        "non_bots": int(label_counts.get(0, 0)),
        "bots": int(label_counts.get(1, 0)),
        "label_plot": label_dist_base64,
        "conf_matrix_plot": conf_matrix_base64,
        "comments": comments
    }
