import json
import os
import re
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from bert import BERTWithMetadata
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Load environment variables (e.g., API keys)
load_dotenv()
YOUTUBE_API_KEY = os.getenv("API-KEY")  # Your YouTube API key
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"

def heuristic_label(comment_dict):
    """
    Determine a bot label for a comment based on its text and engagement.
    Returns 1 if the comment is likely from a bot, 0 otherwise.
    
    Criteria:
      - The text contains spam-related keywords.
      - The comment includes a link (detected via regex) and has zero likes.
    """
    print(comment_dict.keys())
    text = comment_dict["text"].lower()

    score = 0
    spam_patterns = {
        r'\b(?:click|subscribe|free|giveaway|bot)\b': 2,
        r'\b(?:crypto|bitcoin|ethereum|nft|blockchain|dogecoin|altcoin|solana|shiba inu|meme)\b': 3,
        r'(?:check out|visit) (?:my|our) (?:channel|site|website)': 3,
        r'\b(?:win|winner|limited offer|discount|promo)\b': 2
    }
    partition_value = 3
    
    has_link = comment_dict["has_link"] == 1
    has_likes = comment_dict["like_count"] > 0
    has_crypto_address = re.search(r"(0x[a-fA-F0-9]{40}|bc1[a-zA-Z0-9]{39}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})", text)
    
    for pattern, weight in spam_patterns.items():
        if re.search(pattern, text):
            score += weight
            
    if has_crypto_address:
        score += 5
        
    if has_link:
        score += 2
        if not has_likes:
            score += 3
        # shortened links
        if re.search(r"(bit\.ly|goo\.gl|tinyurl|t\.co|ow\.ly)", text):
            score += 2
        # multiple links
        if len(re.findall(r"http[s]?://", text)) > 1:
            score += 1
    
    if comment_dict["like_count"] == 0:
        score += 1
    elif comment_dict["like_count"] > 10:  # Highly liked comments less likely to be spam
        score -= 2
        
    # late night
    if 2 <= comment_dict["published_at"] <= 5: 
        score += 1
      
   
    if len(text) > 150:  # long comments
        score += 1
    if len(text) < 15:   # short comments
        score += 1
    if len(re.findall(r"\!{2,}", text)) > 2:  # multiple exclamation marks
        score += 1
        
    if re.search(r"\b(\w+)\b.*\b\1\b", text):  # rpeated words
        score += 1
    
    # users detected it's bot and gave it a low rating
    if comment_dict["viewer_rating"] < 2 and "great" in text:
        score += 1 
        
    # new channel
    if comment_dict["new_channel"] == 1:
        score += 1
        
    # early comment
    if comment_dict["early_comment"] == 1:
        score += 1
    
    return 1 if score >= partition_value else 0

def parse_timestamp(timestamp_str):
    """Try parsing timestamp with microseconds; if that fails, try without."""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(timestamp_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Timestamp '{timestamp_str}' does not match expected formats.")

def fetch_youtube_comments(video_id, max_results):
    """
    Fetch comments from a specified YouTube video and return a DataFrame.
    
    For each comment:
      - Extract metadata including text, like count, published timestamp (as a numeric feature), and derived features.
      - Check if the comment was made within 60 seconds of the video's publish time.
      - Check if the author's channel is new (created less than 30 days ago).
      - Label as bot (1) if any of these conditions are met, otherwise apply heuristic rules.
      - Write valid comments to "comments.txt" and invalid/excluded comments to "excluded_comments.txt".
    """
    youtube = googleapiclient.discovery.build(API_SERVICE_NAME, API_VERSION, developerKey="AIzaSyAiPecVj0SD41x480zXT-nTVSVtm3RY8rs")
    
    # Get the video's published time.
    video_request = youtube.videos().list(
        part="snippet",
        id=video_id
    )
    video_response = video_request.execute()
    video_publishedAt = video_response["items"][0]["snippet"]["publishedAt"]
    video_published_dt = datetime.strptime(video_publishedAt, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=max_results,
        order="relevance"
    )
    response = request.execute()
    
    # # Rotate log files if they already exist.
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # if os.path.exists("comments.txt"):
    #     os.rename("comments.txt", f"comments_{timestamp}.txt")
    # if os.path.exists("excluded_comments.txt"):
    #     os.rename("excluded_comments.txt", f"excluded_comments_{timestamp}.txt")
    
    comments_file = open("comments.txt", "w", encoding="utf-8")
    excluded_file = open("excluded_comments.txt", "w", encoding="utf-8")
    
    # Cache to avoid duplicate API calls for the same channel.
        # Cache to avoid duplicate API calls for the same channel.
    channel_cache = {}
    comments_data = []
    total_fetched = 0
    next_page_token = None

    def is_new_channel(channel_id):
        """
        Return True if the channel was created less than 30 days ago.
        Uses a cache to prevent duplicate API calls.
        """
        if channel_id in channel_cache:
            return channel_cache[channel_id]
        try:
            channel_request = youtube.channels().list(
                part="snippet",
                id=channel_id
            )
            channel_response = channel_request.execute()
            if not channel_response.get("items"):
                result = False
            else:
                publishedAt = channel_response["items"][0]["snippet"]["publishedAt"]
                published_date = parse_timestamp(publishedAt)
                now = datetime.now(timezone.utc)
                result = (now - published_date) < timedelta(days=180)
            channel_cache[channel_id] = result
            return result
        except Exception as e:
            print(f"Error fetching channel {channel_id}: {e}")
            channel_cache[channel_id] = False
            return False

    while total_fetched < max_results:
        # Set maxResults per request to the minimum of 100 or remaining comments to fetch.
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=min(100, max_results - total_fetched),
            order="relevance",
            pageToken=next_page_token
        )
        response = request.execute()
        items = response.get("items", [])
        if not items:
            break  # No more comments available.
        
        for item in tqdm(items, desc="Parsing Comments", leave=False):
            snippet = item['snippet']['topLevelComment']['snippet']
            text = snippet.get("textDisplay", "")
            like_count = snippet.get("likeCount", 0)
            
            # Exclude comments with empty text.
            if not text.strip():
                excluded_file.write(json.dumps({"reason": "empty text", "data": snippet}, indent=4))
                excluded_file.write("\n\n--- NEW COMMENT ---\n\n")
                continue
            
            # Extract comment published time as a string and as a datetime.
            comment_published_str = snippet.get("publishedAt", "")
            try:
                comment_published_dt = datetime.strptime(comment_published_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                # Convert to numeric feature (seconds since epoch).
                published_at_numeric = comment_published_dt.timestamp()
            except Exception as e:
                excluded_file.write(json.dumps({"reason": f"timestamp parse error: {e}", "data": snippet}, indent=4))
                excluded_file.write("\n\n--- NEW COMMENT ---\n\n")
                continue
            
            # Use regex to detect if the comment contains a link.
            has_link = 1 if re.search(r'https?://\S+', text) else 0
            
            # Optional additional metadata.
            viewer_rating = 0     # Placeholder; add logic if available.
            
            # Extract the author's channel ID.
            channel_info = snippet.get("authorChannelId", {})
            channel_id = channel_info.get("value")
            
            # Build the comment dictionary.
            comment_dict = {
                'text': text,
                'like_count': like_count,
                'published_at': published_at_numeric,  # Numeric timestamp feature
                'viewer_rating': viewer_rating,
                'has_link': has_link,
            }
            
            # Determine label:
            # 1. Early comment check: if posted within 60 seconds of video publish time.
            if comment_published_dt and (comment_published_dt - video_published_dt) < timedelta(seconds=60):
                comment_dict['early_comment'] = 1
                comment_dict['label'] = 1
            # 2. New channel check: if author's channel is new (< 6 months old).
            elif channel_id and is_new_channel(channel_id):
                comment_dict['new_channel'] = 1
                comment_dict['label'] = 1
            else:
                comment_dict['early_comment'] = 0
                comment_dict['new_channel'] = 0
                comment_dict['label'] = heuristic_label(comment_dict)
            
            comments_data.append(comment_dict)
            comments_file.write(json.dumps(comment_dict, indent=4))
            comments_file.write("\n\n--- NEW COMMENT ---\n\n")
            total_fetched += 1
        
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break  # No more pages.
    
    comments_file.close()
    excluded_file.close()
    
    df = pd.DataFrame(comments_data)
    
    # DataFrame Check: if no valid comments are found, log and return an empty DataFrame.
    if df.empty:
        print("No valid comments found for training. Exiting.")
        return df
    
    return df

# Fetch data from YouTube for a specific video.
video_id = "aKq8bkY5eTU"  # I Survived The 5 Deadliest Places On Earth
data = fetch_youtube_comments(video_id, max_results=5000)

# DataFrame Check: Exit if no valid comments.
if data.empty:
    print("No valid comments to train on. Exiting program.")
    exit()

print(data.head())

# Preprocess metadata features.
# Now we include 'published_at' (numeric timestamp) as a training feature.
meta_features = ['like_count', 'published_at', 'viewer_rating', 'has_link']
scaler = StandardScaler()
meta_scaled = scaler.fit_transform(data[meta_features])
meta_tensor = torch.tensor(meta_scaled, dtype=torch.float)

# Tokenize text using the DistilBertTokenizer.
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
encodings = tokenizer(list(data['text']), padding=True, truncation=True, return_tensors='pt')

# Prepare labels.
labels = torch.tensor(data['label'].values).float().unsqueeze(1)

# Create a custom Dataset class.
class YouTubeCommentDataset(Dataset):
    def __init__(self, encodings, metadata, labels):
        self.encodings = encodings
        self.metadata = metadata
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'metadata': self.metadata[idx],
            'label': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

dataset = YouTubeCommentDataset(encodings, meta_tensor, labels)
loader = DataLoader(dataset, batch_size=2, shuffle=True)


# Initialize your BERT-based model.
model = BERTWithMetadata(metadata_dim=meta_tensor.shape[1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up loss function and optimizer.
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Training loop (for demonstration).
num_epochs = 3
model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        metadata = batch['metadata'].to(device)
        labels_batch = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, metadata)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        print(f"Batch Loss: {loss.item():.4f}")
    print("----- End of Epoch -----\n")


# Switch to evaluation mode
model.eval()

all_preds = []
all_labels = []

# Disable gradient calculations for inference
with torch.no_grad():
    for batch in loader:  # ideally a test or validation DataLoader
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        metadata = batch['metadata'].to(device)
        labels_batch = batch['label'].to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask, metadata)
        
        # Convert outputs to binary predictions using a threshold (e.g., 0.5)
        preds = (outputs > 0.5).float()
        
        # Collect predictions and true labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

# Convert lists to numpy arrays and reshape
all_preds = np.array(all_preds).reshape(-1)
all_labels = np.array(all_labels).reshape(-1)

# Compute and print various metrics
print("Accuracy:", accuracy_score(all_labels, all_preds))
print("Precision:", precision_score(all_labels, all_preds, zero_division=0))
print("Recall:", recall_score(all_labels, all_preds, zero_division=0))
print("F1 Score:", f1_score(all_labels, all_preds, zero_division=0))

# Print a detailed classification report
print("\nClassification Report:\n", classification_report(all_labels, all_preds, zero_division=0))