import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Model_RNN import (
    build_vocabulary, clean_text, extract_features, CommentDataset,
    BotCommentClassifier, MAX_LEN, feature_names
)
from Trained_bert import fetch_api_comments, save_plot_to_base64, youtube_api_request

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inference function
# --- Model and vocab loaded ONCE at module level ---
RNN_MODEL_PATH = "best_RNN_model.pth"

# Load checkpoint and vocab size once
_rnn_checkpoint = torch.load(RNN_MODEL_PATH, map_location=device)
_rnn_vocab_size = _rnn_checkpoint['embedding.weight'].shape[0]

# Initialize and load model once
_rnn_model = BotCommentClassifier(
    vocab_size=_rnn_vocab_size,
    embedding_dim=200,
    hidden_dim=256,
    num_features=len(feature_names),
    dropout_rate=0.3
)
_rnn_model.load_state_dict(_rnn_checkpoint)
_rnn_model.to(device)
_rnn_model.eval()

# --- Comment-level cache (in-memory, per session) ---
_rnn_comment_cache = {}


def run_rnn_inference_on_video(video_id, max_results=5000):
    # Get video title
    try:
        video_response = youtube_api_request("videos", {"part": "snippet", "id": video_id})
        items = video_response.get('items', [])
        video_title = items[0]['snippet']['title'] if items else "Unknown Title"
    except Exception as e:
        print(f"Error fetching video info: {e}")
        video_title = "Unknown Title"

    # Fetch and clean data
    df = fetch_api_comments(video_id, max_results)
    df['CONTENT'] = df['text'].apply(clean_text)

    # Build vocab based on current input (for tokenization only)
    vocab = build_vocabulary(df)

    # Prepare features
    features = df['CONTENT'].apply(extract_features)
    for i, name in enumerate(feature_names):
        df[name] = [f[i] for f in features]

    # --- Efficient Preprocessing (vectorized) ---
    # Check comment-level cache
    comment_results = []
    comments_to_process = []
    indices_to_process = []
    for idx, row in df.iterrows():
        comment_text = row['CONTENT']
        cache_key = (video_id, comment_text)
        if cache_key in _rnn_comment_cache:
            comment_results.append(_rnn_comment_cache[cache_key])
        else:
            comment_results.append(None)
            comments_to_process.append((idx, row))
            indices_to_process.append(idx)

    # Only process uncached comments
    if comments_to_process:
        temp_df = df.loc[indices_to_process].copy()
        temp_vocab = build_vocabulary(temp_df)
        temp_dataset = CommentDataset(temp_df, temp_vocab, MAX_LEN, feature_names)
        temp_loader = DataLoader(temp_dataset, batch_size=32, shuffle=False)
        predictions = []
        with torch.no_grad():
            for batch in temp_loader:
                text = batch['text'].to(device)
                features = batch['features'].to(device) if batch['features'] is not None else None
                outputs = _rnn_model(text, features)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                predictions.extend(preds.cpu().numpy())
        # Store results in cache and update comment_results
        for idx2, pred in enumerate(predictions):
            comment_row = comments_to_process[idx2][1]
            cache_key = (video_id, comment_row['CONTENT'])
            pred_label = int(pred)
            _rnn_comment_cache[cache_key] = pred_label
            comment_results[indices_to_process[idx2]] = pred_label

    # Now, comment_results has all predictions in order
    df['predicted_label'] = comment_results
    label_counts = df['predicted_label'].value_counts().sort_index()

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(label_counts.index.astype(str), label_counts.values, color=['green', 'red'])
    ax.set_xlabel("Predicted Label (0 = non-bot, 1 = bot)")
    ax.set_ylabel("Number of Comments")
    ax.set_title("Distribution of Predicted Labels (RNN)")
    plt.tight_layout()
    label_plot_base64 = save_plot_to_base64(fig)
    plt.close(fig)

    # Build enriched comments array
    comments = []
    for idx, row in df.iterrows():
        comment_obj = {
            'id': str(idx),
            'text': row.get('text', row.get('CONTENT', '')),
            'modelResults': {
                'rnnModel': {
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