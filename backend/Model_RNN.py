import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import re
from collections import Counter

feature_names = ['caps_ratio', 'repeated_chars', 'word_count', 'avg_word_length']

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define hyperparameters
MAX_LEN = 100
EMBEDDING_DIM = 200
HIDDEN_DIM = 256
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
EARLY_STOPPING_PATIENCE = 3
CLASS_WEIGHTS = [0.6, 1.4]  # More weight on bot class

# Basic text preprocessing
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+', ' URL ', text)
        text = re.sub(r'@\w+', ' MENTION ', text)
        text = re.sub(r'#\w+', ' HASHTAG ', text)
        text = re.sub(r'\d+', ' NUMBER ', text)
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    else:
        return ''

# Extract features that might indicate bot behavior
def extract_features(text):
    if not isinstance(text, str) or not text:
        return [0, 0, 0, 0]
    
    caps_ratio = sum(1 for c in text if c.isupper()) / (len(text) + 1)
    repeated_chars = len(re.findall(r'(.)\1{2,}', text))
    word_count = len(text.split())
    avg_word_length = sum(len(word) for word in text.split()) / (word_count + 1)
    
    return [caps_ratio, repeated_chars, word_count, avg_word_length]

# Build vocabulary with unigrams and bigrams
def build_vocabulary(dataframe, max_vocab_size=8000):
    all_tokens = []
    
    for text in dataframe['CONTENT']:
        tokens = text.split()
        all_tokens.extend(tokens)  # Unigrams
        
        # Add bigrams
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        all_tokens.extend(bigrams)
        
    token_counter = Counter(all_tokens)
    vocabulary = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in token_counter.most_common(max_vocab_size - len(vocabulary)):
        vocabulary[word] = len(vocabulary)
    
    print(f"Vocabulary size: {len(vocabulary)}")
    return vocabulary

# Dataset class to handle text and features
class CommentDataset(Dataset):
    def __init__(self, dataframe, vocab, max_len, feature_names=None):
        self.dataframe = dataframe
        self.vocab = vocab
        self.max_len = max_len
        self.feature_names = feature_names if feature_names else []

    def __len__(self):
        return len(self.dataframe)
    
    def tokenize_with_ngrams(self, text):
        tokens = text.split()
        unigrams = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Add bigram indices
        bigram_indices = []
        if len(tokens) > 1:
            for i in range(len(tokens)-1):
                bigram = f"{tokens[i]}_{tokens[i+1]}"
                bigram_indices.append(self.vocab.get(bigram, self.vocab['<UNK>']))
        
        # Combine and pad/truncate to max_len
        combined = unigrams + bigram_indices
        if len(combined) > self.max_len:
            combined = combined[:self.max_len]
        else:
            combined += [self.vocab['<PAD>']] * (self.max_len - len(combined))
        
        return combined

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['CONTENT']
        label = self.dataframe.iloc[idx].get('label', -1)
        
        indices = self.tokenize_with_ngrams(text)
        
        extra_features = []
        if self.feature_names:
            extra_features = [self.dataframe.iloc[idx][feat] for feat in self.feature_names]
            
        return {
            'text': torch.tensor(indices, dtype=torch.long),
            'features': torch.tensor(extra_features, dtype=torch.float) if extra_features else None,
            'label': torch.tensor(label, dtype=torch.long)
        }

# Define the model with attention
class BotCommentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_features, output_dim=2, dropout_rate=0.3):
        super(BotCommentClassifier, self).__init__()
        
        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim*2, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc = nn.Linear(hidden_dim*2 + num_features, output_dim)
        
    def attention_mechanism(self, lstm_output):
        # Apply attention to get weighted sum of hidden states
        attn_weights = self.attention(lstm_output)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context
        
    def forward(self, text, features=None):
        # Get embeddings
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)
        
        # Pass through LSTM layer
        lstm_out, _ = self.lstm(embedded)
        
        # Apply attention
        context = self.attention_mechanism(lstm_out)
        
        # Concatenate with additional features if available
        if features is not None:
            combined = torch.cat((context, features), dim=1)
        else:
            combined = context
        
        # Final classification
        output = self.fc(combined)
        
        return output