import os
import json
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, AdamW
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from bert import BERTWithMetadata
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load environment variables (e.g., API keys)
load_dotenv()

##############################
# Helper Functions (same as API version)
##############################

def parse_timestamp(timestamp_str):
    """Try multiple formats to parse a timestamp string into a datetime."""
    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Timestamp '{timestamp_str}' does not match expected formats.")

def heuristic_label(comment_dict):
    """
    Determine a bot label for a comment based on its text and engagement.
    Returns 1 if the comment is likely from a bot, 0 otherwise.
    
    Criteria:
      - The text contains spam-related keywords.
      - The comment includes a link (detected via regex) and has zero likes.
    """
    text = comment_dict["text"].lower()
    spam_keywords = ["click", "subscribe", "free", "crypto", "giveaway", "bot"]
    
    if any(keyword in text for keyword in spam_keywords):
        return 1
    if comment_dict["has_link"] == 1 and comment_dict["like_count"] == 0:
        return 1
    return 0

##############################
# CSV Processing Function
##############################

def fetch_csv_comments(csv_file):
    # Read CSV and remove extra whitespace from column names.
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    print("Original CSV rows:", len(df))
    
    # Rename columns for consistency.
    df = df.rename(columns={'CONTENT': 'text', 'DATE': 'date', 'CLASS': 'label'})
    
    # Drop rows with missing text or label.
    df = df.dropna(subset=['text', 'label'])
    print("Rows after dropna:", len(df))
    
    # Determine video publish time as the earliest valid date in the CSV.
    valid_dates = df['date'].dropna().unique()
    parsed_dates = []
    for d in valid_dates:
        if pd.isna(d) or not isinstance(d, str) or d.strip() == "":
            continue
        try:
            parsed_dates.append(parse_timestamp(d))
        except Exception as e:
            print(f"Date parse error for {d}: {e}")
    if parsed_dates:
        video_published_timestamp = min([dt.timestamp() for dt in parsed_dates])
        video_published_dt = datetime.fromtimestamp(video_published_timestamp, tz=timezone.utc)
        print("Assumed video publish time:", video_published_dt)
    else:
        video_published_dt = None

    # Rotate log files if they already exist.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if os.path.exists("commentsFor2.txt"):
        os.rename("commentsFor2.txt", f"commentsFor2_{timestamp}.txt")
    if os.path.exists("excluded_commentsFor2.txt"):
        os.rename("excluded_commentsFor2.txt", f"excluded_commentsFor2_{timestamp}.txt")
    comments_file = open("commentsFor2.txt", "w", encoding="utf-8")
    excluded_file = open("excluded_commentsFor2.txt", "w", encoding="utf-8")
    
    comments_data = []
    for _, row in df.iterrows():
        text = row['text']
        if not text.strip():
            excluded_file.write(json.dumps({"reason": "empty text", "data": row.to_dict()}, indent=4))
            excluded_file.write("\n\n--- NEW COMMENT ---\n\n")
            continue

        date_val = row.get('date', "")
        if pd.isna(date_val) or not isinstance(date_val, str) or date_val.strip() == "":
            published_at_numeric = 0.0
            comment_dt = None
        else:
            try:
                comment_dt = parse_timestamp(date_val)
                published_at_numeric = comment_dt.timestamp()
            except Exception as e:
                excluded_file.write(json.dumps({"reason": f"timestamp parse error: {e}", "data": row.to_dict()}, indent=4))
                excluded_file.write("\n\n--- NEW COMMENT ---\n\n")
                continue
        
        # Detect if the comment contains a link.
        has_link = 1 if re.search(r'https?://\S+', text) else 0
        
        # For CSV, we assume like_count and viewer_rating are 0.
        like_count = 0
        viewer_rating = 0

        # Build comment dictionary.
        comment_dict = {
            'text': text,
            'like_count': like_count,
            'published_at': published_at_numeric,
            'viewer_rating': viewer_rating,
            'has_link': has_link,
        }
        
        # Check if comment was posted within 60 seconds of video publish time.
        if video_published_dt and comment_dt:
            if (comment_dt - video_published_dt) < timedelta(seconds=60):
                comment_dict['early_comment'] = 1
            else:
                comment_dict['early_comment'] = 0
        else:
            comment_dict['early_comment'] = 0
        
        # No channel info from CSV; set new_channel to 0.
        comment_dict['new_channel'] = 0
        
        # Use the ground truth label from CSV.
        comment_dict['label'] = int(row['label'])
        
        # Optionally, compute the heuristic label (for debugging/comparison).
        comment_dict['heuristic'] = heuristic_label(comment_dict)
        
        comments_data.append(comment_dict)
        comments_file.write(json.dumps(comment_dict, indent=4))
        comments_file.write("\n\n--- NEW COMMENT ---\n\n")
    
    # Close log files.
    comments_file.close()
    excluded_file.close()
    
    df_processed = pd.DataFrame(comments_data)
    print("Rows after processing:", len(df_processed))
    
    if df_processed.empty:
        print("No valid comments found for training. Exiting.")
    
    return df_processed

##############################
# Main Training Program
##############################

# Use the CSV dataset as if it were the API response.
data = fetch_csv_comments("Youtube04-Eminem.csv")

if data.empty:
    print("No valid comments to train on. Exiting program.")
    exit()

print("Processed DataFrame head:")
print(data.head())

# Preprocess metadata features.
# Here we use 'like_count', 'published_at', 'viewer_rating', and 'has_link' as metadata features.
meta_features = ['like_count', 'published_at', 'viewer_rating', 'has_link']
scaler = StandardScaler()
meta_scaled = scaler.fit_transform(data[meta_features])
meta_tensor = torch.tensor(meta_scaled, dtype=torch.float)

# Save the scaler to scaler.pkl for later use during inference.
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Scaler saved as scaler.pkl.")

# Tokenize text.
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
encodings = tokenizer(list(data['text']), padding=True, truncation=True, return_tensors='pt')

# Prepare labels.
labels = torch.tensor(data['label'].values).float().unsqueeze(1)

# Custom Dataset.
class YoutubeCSVCommentDataset(Dataset):
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

dataset = YoutubeCSVCommentDataset(encodings, meta_tensor, labels)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize the model.
metadata_dim = meta_tensor.shape[1]
model = BERTWithMetadata(metadata_dim=metadata_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and Optimizer.
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Training loop.
num_epochs = 3
epoch_losses = []      # To store average loss per epoch.
epoch_accuracies = []  # To store training accuracy per epoch.
model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    running_loss = 0.0
    batch_count = 0
    correct = 0
    total = 0
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
        
        running_loss += loss.item()
        batch_count += 1
        
        # Compute predictions and accumulate correct counts.
        preds = (outputs > 0.5).float()
        correct += (preds == labels_batch).sum().item()
        total += labels_batch.size(0)
        
        print(f"Batch Loss: {loss.item():.4f}")
    epoch_avg_loss = running_loss / batch_count
    epoch_accuracy = correct / total if total > 0 else 0
    epoch_losses.append(epoch_avg_loss)
    epoch_accuracies.append(epoch_accuracy)
    print(f"Epoch {epoch+1} Average Loss: {epoch_avg_loss:.4f}")
    print(f"Epoch {epoch+1} Training Accuracy: {epoch_accuracy:.4f}")
    print("----- End of Epoch -----\n")

# Save the trained model.
torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth.")

# Define a function to plot the learning curve.
def plot_learning_curve(losses, accuracies):
    epochs = range(1, len(losses) + 1)
    plt.figure(figsize=(12, 5))
    
    # Plot training loss.
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, marker='o', linestyle='-', color='blue')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    plt.grid(True)
    
    # Plot training accuracy.
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, marker='o', linestyle='-', color='green')
    plt.title("Training Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(epochs)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("learning_curve.png")
    plt.show()
    print("Learning curve plot saved as learning_curve.png.")

# Call the function to display the learning curve.
plot_learning_curve(epoch_losses, epoch_accuracies)
# Plot training loss over epochs.
plt.figure(figsize=(8, 4))
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', linestyle='-', color='blue')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(range(1, num_epochs + 1))
plt.grid(True)
plt.savefig("training_loss.png")
plt.show()
print("Training loss plot saved as training_loss.png.")

# Evaluation.
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        metadata = batch['metadata'].to(device)
        labels_batch = batch['label'].to(device)

        outputs = model(input_ids, attention_mask, metadata)
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

all_preds = np.array(all_preds).reshape(-1)
all_labels = np.array(all_labels).reshape(-1)

# Overall Metrics.
overall_accuracy = accuracy_score(all_labels, all_preds)
overall_precision = precision_score(all_labels, all_preds, zero_division=0)
overall_recall = recall_score(all_labels, all_preds, zero_division=0)
overall_f1 = f1_score(all_labels, all_preds, zero_division=0)

print("Overall Accuracy:", overall_accuracy)
print("Overall Precision:", overall_precision)
print("Overall Recall:", overall_recall)
print("Overall F1 Score:", overall_f1)

# Detailed Classification Report.
report = classification_report(all_labels, all_preds, zero_division=0)
print("\nClassification Report:\n", report)

# Plot the confusion matrix.
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
print("Confusion matrix plot saved as confusion_matrix.png.")

# Extract metrics for the bot class (label 1).
bot_precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
bot_recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
bot_f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)

print("\nBot Class Metrics:")
print("Bot Precision (when predicted as bot, correct %):", bot_precision)
print("Bot Recall (coverage of actual bots):", bot_recall)
print("Bot F1 Score:", bot_f1)
