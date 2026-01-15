import os
import json
import re
import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from bert import BERTWithMetadata
from datetime import datetime, timezone
import googleapiclient.discovery
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load environment variables if needed.
load_dotenv()
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"
YOUTUBE_API_KEY = os.getenv("API-KEY")  # or set directly: "YOUR_YOUTUBE_API_KEY"

# Build YouTube API client.
youtube = googleapiclient.discovery.build(API_SERVICE_NAME, API_VERSION, developerKey="AIzaSyAiPecVj0SD41x480zXT-nTVSVtm3RY8rs")

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

def fetch_api_comments(video_id, max_results):
    """
    Fetch comments from the YouTube API for a given video.
    Returns a DataFrame with the following columns:
      - text: comment text
      - like_count: number of likes (if available)
      - published_at: numeric timestamp (seconds since epoch)
      - viewer_rating: defaulted to 0
      - has_link: 1 if comment contains a URL, else 0
    Also logs valid comments into a log file and excluded ones into another.
    This implementation fetches multiple pages until up to max_results comments are collected.
    """
    # Prepare log files.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    valid_log_filename = f"commentsForAPI_{timestamp}.txt"
    excluded_log_filename = f"excluded_commentsForAPI_{timestamp}.txt"
    valid_file = open(valid_log_filename, "w", encoding="utf-8")
    excluded_file = open(excluded_log_filename, "w", encoding="utf-8")
    
    comments = []
    nextPageToken = None
    
    # Loop until we've fetched the desired number of comments or no more pages exist.
    while len(comments) < max_results:
        # YouTube API maxResults is limited to 100 per request.
        current_max = min(100, max_results - len(comments))
        request_params = {
            "part": "snippet",
            "videoId": video_id,
            "textFormat": "plainText",
            "maxResults": current_max,
            "order": "relevance"
        }
        if nextPageToken:
            request_params["pageToken"] = nextPageToken
        
        request = youtube.commentThreads().list(**request_params)
        response = request.execute()
        
        for item in tqdm(response.get("items", []), desc="Fetching API Comments", leave=False):
            snippet = item['snippet']['topLevelComment']['snippet']
            text = snippet.get("textDisplay", "")
            # Exclude if text is empty.
            if not text.strip():
                excluded_file.write(json.dumps({"reason": "empty text", "data": snippet}, indent=4))
                excluded_file.write("\n\n--- NEW COMMENT ---\n\n")
                continue

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
                continue

            viewer_rating = 0  # Default value.
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
            
            if len(comments) >= max_results:
                break  # Exit early if reached desired count

        nextPageToken = response.get("nextPageToken")
        if not nextPageToken:
            break  # No further pages available

    valid_file.close()
    excluded_file.close()
    print(f"Valid comments logged to {valid_log_filename}")
    print(f"Excluded comments logged to {excluded_log_filename}")
    
    return pd.DataFrame(comments)

# Specify the video ID for inference.
video_id = "FsNnI_ynqG0"  # Playing Repo With The boys (Jynxzi And Sketch) CaseOh

# Fetch comments (fetch_api_comments function remains unchanged)
new_comments_df = fetch_api_comments(video_id, max_results=3000)
print("Fetched new comments:", new_comments_df.shape[0])

# Preprocess the metadata.
meta_features = ['like_count', 'published_at', 'viewer_rating', 'has_link']
meta_data = new_comments_df[meta_features]
meta_scaled = scaler.transform(meta_data)
meta_tensor = torch.tensor(meta_scaled, dtype=torch.float)

# Tokenize the text.
encodings = tokenizer(list(new_comments_df['text']), padding=True, truncation=True, return_tensors='pt')

# Create the inference dataset.
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

inference_dataset = InferenceDataset(encodings, meta_tensor)
inference_loader = torch.utils.data.DataLoader(inference_dataset, batch_size=8, shuffle=False)

# Run inference with progress tracking.
print("Starting inference...")
all_preds = []
with torch.no_grad():
    for batch in tqdm(inference_loader, desc="Running inference"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        metadata = batch['metadata'].to(device)
        
        outputs = model(input_ids, attention_mask, metadata)
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())

print("Inference completed.")

all_preds = np.array(all_preds).reshape(-1)
new_comments_df['predicted_label'] = all_preds
print("Inference Results:")
print(new_comments_df.head())


# Plot a bar chart of predicted label distribution.
label_counts = new_comments_df['predicted_label'].value_counts().sort_index()
plt.figure(figsize=(6, 4))
plt.bar(label_counts.index.astype(str), label_counts.values, color=['green', 'red'])
plt.xlabel("Predicted Label (0 = non-bot, 1 = bot)")
plt.ylabel("Number of Comments")
plt.title("Distribution of Predicted Labels")
plt.savefig("predicted_label_distribution.png")
plt.show()
print("Predicted label distribution plot saved as predicted_label_distribution.png.")

# Optionally, if you have ground truth labels for these comments, plot a confusion matrix.
if 'label' in new_comments_df.columns and new_comments_df['label'].nunique() > 1:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(new_comments_df['label'].values, all_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("inference_confusion_matrix.png")
    plt.show()
    print("Confusion matrix plot saved as inference_confusion_matrix.png.")
