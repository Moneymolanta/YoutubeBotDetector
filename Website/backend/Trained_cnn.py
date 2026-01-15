import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import re
from Model_CNN import CNN
from Trained_bert import fetch_api_comments, save_plot_to_base64, youtube_api_request

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    return text.strip()

def tokenize_text(text):
    return text.split()

def build_vocab_from_comments(text_list, max_vocab_size=10000):
    all_tokens = []
    for text in text_list:
        tokens = tokenize_text(text)
        all_tokens.extend(tokens)
    token_counter = Counter(all_tokens)
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(token_counter.most_common(max_vocab_size))}
    return vocab

class CommentDataset(Dataset):
    def __init__(self, df, vocab, max_len=100):
        self.df = df
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["text"]
        tokens = tokenize_text(text)
        tokens = [self.vocab.get(token, 0) for token in tokens]
        tokens = tokens[:self.max_len] + [0] * (self.max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)


# --- Model and vocab loaded ONCE at module level ---
CNN_MODEL_PATH = "best_CNN_model.pth"
CNN_VOCAB_PATH = "cnn_vocab.txt"

# Load vocab once
_cnn_vocab = {}
with open(CNN_VOCAB_PATH, "r") as f:
    for idx, line in enumerate(f):
        token = line.strip()
        _cnn_vocab[token] = idx + 1  # 1-based index, 0 is UNK

# Set vocab_size to match the vocab file (number of tokens + 1 for UNK)
_cnn_vocab_size = len(_cnn_vocab) + 1

# Load model once
_cnn_model = CNN(batch_size=32, embedding_dim=100, output_dim=2, vocab_size=_cnn_vocab_size)
_cnn_checkpoint = torch.load(CNN_MODEL_PATH, map_location=device)
_cnn_model.load_state_dict(_cnn_checkpoint)
_cnn_model.to(device)
_cnn_model.eval()

# --- Comment-level cache (in-memory, per session) ---
_cnn_comment_cache = {}


def run_cnn_inference_on_video(video_id, max_results=5000):
    # Get video title
    try:
        video_response = youtube_api_request("videos", {"part": "snippet", "id": video_id})
        items = video_response.get('items', [])
        video_title = items[0]['snippet']['title'] if items else "Unknown Title"
    except Exception as e:
        print(f"Error fetching video info: {e}")
        video_title = "Unknown Title"

    # Fetch and clean comments
    df = fetch_api_comments(video_id, max_results)
    if df is None or df.empty or 'text' not in df.columns:
        print(f"[CNN] No comments or missing 'text' column for video_id={video_id}")
        return {
            "error": "No comments fetched or missing 'text' column. Check YouTube API key, quota, or video availability."
        }
    df['text'] = df['text'].apply(clean_text)

    # --- Efficient Preprocessing (vectorized) ---
    # Check comment-level cache
    comment_results = []
    comments_to_process = []
    indices_to_process = []
    for idx, row in df.iterrows():
        comment_text = row['text']
        cache_key = (video_id, comment_text)
        if cache_key in _cnn_comment_cache:
            comment_results.append(_cnn_comment_cache[cache_key])
        else:
            comment_results.append(None)
            comments_to_process.append(comment_text)
            indices_to_process.append(idx)

    # Only process uncached comments
    if comments_to_process:
        temp_df = df.loc[indices_to_process].copy()
        temp_dataset = CommentDataset(temp_df, _cnn_vocab)
        temp_dataloader = DataLoader(temp_dataset, batch_size=32, shuffle=False, drop_last=False)
        predictions = []
        with torch.no_grad():
            for batch in temp_dataloader:
                batch = batch.to(device)
                outputs = _cnn_model(batch)
                _, predicted = torch.max(outputs, dim=1)
                predictions.extend(predicted.cpu().numpy())
        # Store results in cache and update comment_results
        for idx2, pred in enumerate(predictions):
            cache_key = (video_id, comments_to_process[idx2])
            pred_label = int(pred)
            _cnn_comment_cache[cache_key] = pred_label
            comment_results[indices_to_process[idx2]] = pred_label

    # Now, comment_results has all predictions in order
    df['predicted_label'] = comment_results
    label_counts = df['predicted_label'].value_counts().sort_index()

    # Bar Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(label_counts.index.astype(str), label_counts.values, color=['green', 'red'])
    ax.set_xlabel("Predicted Label (0 = non-bot, 1 = bot)")
    ax.set_ylabel("Number of Comments")
    ax.set_title("Distribution of Predicted Labels (CNN)")
    plt.tight_layout()
    label_plot_base64 = save_plot_to_base64(fig)
    plt.close(fig)

    # Build enriched comments array
    comments = []
    for idx, row in df.iterrows():
        comment_obj = {
            'id': str(idx),
            'text': row.get('text', ''),
            'modelResults': {
                'cnnModel': {
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
        "total": int(len(df)),
        "non_bots": int(label_counts.get(0, 0)),
        "bots": int(label_counts.get(1, 0)),
        "label_plot": label_plot_base64,
        "conf_matrix_plot": None,
        "comments": comments
    }
