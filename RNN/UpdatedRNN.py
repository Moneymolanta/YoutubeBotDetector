import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define hyperparameters
MAX_LEN = 100
EMBEDDING_DIM = 200
HIDDEN_DIM = 192
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
EARLY_STOPPING_PATIENCE = 3
CLASS_WEIGHTS = [0.6, 1.4]  # More weight on bot class

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/Youtube Spam Dataset/Youtube-Spam-Dataset.csv')

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

# Clean and preprocess data
print("Preprocessing data...")
df['CONTENT'] = df['CONTENT'].apply(clean_text)
df['label'] = df['CLASS'].astype(int)

# Extract additional features
feature_names = ['caps_ratio', 'repeated_chars', 'word_count', 'avg_word_length']
features = df['CONTENT'].apply(extract_features)
for i, name in enumerate(feature_names):
    df[name] = [f[i] for f in features]

# Remove empty comments
df = df[df['CONTENT'].str.strip() != '']
df = df.reset_index(drop=True)

# Print data insights
class_counts = df['label'].value_counts()
print(f"Class distribution - Human: {class_counts.get(0, 0)}, Bot: {class_counts.get(1, 0)}")

# Split data: 80% train, 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Build vocabulary with unigrams and bigrams
def build_vocabulary(dataframe, max_vocab_size=5000):
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

# Build vocabulary
vocab = build_vocabulary(train_df)
vocab_size = len(vocab)

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

# Create datasets and dataloaders
train_dataset = CommentDataset(train_df, vocab, MAX_LEN, feature_names)
test_dataset = CommentDataset(test_df, vocab, MAX_LEN, feature_names)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

# Initialize model, optimizer, and loss function
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Mac GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
else:
    device = torch.device("cpu")  # CPU fallback

model = BotCommentClassifier(
    vocab_size, 
    EMBEDDING_DIM, 
    HIDDEN_DIM, 
    num_features=len(feature_names), 
    dropout_rate=DROPOUT_RATE
)
model.to(device)

# Use class weights to handle imbalance
class_weights = torch.FloatTensor(CLASS_WEIGHTS).to(device) if CLASS_WEIGHTS else None
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model
print("Training model...")
accuracy_history = []
test_accuracy_history = []
loss_history = []
epoch_numbers = []
best_accuracy = 0
patience_counter = 0

auc_per_epoch = []
precision_per_epoch = []
f1_per_epoch = []
recall_per_epoch = []
best_epoch = 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        text = batch['text'].to(device)
        features = batch['features'].to(device) if batch['features'] is not None else None
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(text, features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Calculate accuracy
    accuracy = correct / total
    accuracy_history.append(100 * accuracy)
    avg_loss = epoch_loss / len(train_loader)
    all_labels = []
    all_preds_epoch = []
    all_probs_epoch = []

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch in train_loader:
            text = batch['text'].to(device)
            features = batch['features'].to(device) if batch['features'] is not None else None
            labels = batch['label'].to(device)
            
            outputs = model(text, features)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds_epoch.extend(predicted.cpu().numpy())
            all_probs_epoch.extend(probs[:, 1].cpu().numpy())

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = test_correct / test_total
    test_accuracy_history.append(100 * test_accuracy)

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_epoch = epoch
        # Save the best model
        torch.save(model.state_dict(), 'best_RNN_model.pth')
        print("Saved best model checkpoint.")

    # Compute metrics
    precision = precision_score(all_labels, all_preds_epoch, zero_division=0)
    recall = recall_score(all_labels, all_preds_epoch, zero_division=0)
    f1 = f1_score(all_labels, all_preds_epoch, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs_epoch)

    auc_per_epoch.append(roc_auc)
    precision_per_epoch.append(precision)
    f1_per_epoch.append(f1)
    recall_per_epoch.append(recall)
    # Store for plotting

    loss_history.append(avg_loss)
    epoch_numbers.append(epoch + 1)
    
    # Store for plotting
    
    # loss_history.append(avg_loss)
    # epoch_numbers.append(epoch + 1)

    # Print everything
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
    print(f"Train Accuracy: {100*accuracy:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f} | ROC AUC: {roc_auc:.4f}")

    # avg_loss = epoch_loss / len(train_loader)
    # # print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Accuracy: {100*accuracy:.2f}%")
    # print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}%")
    
    # # Test set evaluation
    # model.eval()
    # test_correct = 0
    # test_total = 0


    # with torch.no_grad():
    #     for batch in test_loader:
    #         text = batch['text'].to(device)
    #         features = batch['features'].to(device) if batch['features'] is not None else None
    #         labels = batch['label'].to(device)
    #
    #         outputs = model(text, features)
    #         _, predicted = torch.max(outputs, 1)
    #
    #         test_total += labels.size(0)
    #         test_correct += (predicted == labels).sum().item()
    #
    # test_accuracy = test_correct / test_total
    # test_accuracy_history.append(100 * test_accuracy)
    # print(f"Test Accuracy: {100*test_accuracy:.2f}%")
    
    # Early stopping
    # if test_accuracy > best_accuracy:
    #     best_accuracy = test_accuracy
    #     patience_counter = 0
    #     # Save the best model
    #     torch.save(model.state_dict(), 'best_RNN_model.pth')
    #     print("Saved best model checkpoint.")
    # else:
    #     patience_counter += 1
    #     if patience_counter >= EARLY_STOPPING_PATIENCE:
    #         print(f"Early stopping triggered after {epoch+1} epochs")
    #         break

# Load the best model
model.load_state_dict(torch.load('best_RNN_model.pth'))

# Dictionary to store prediction results
prediction_results = {}

# Function to predict and calculate percentage of bot comments
def predict_bot_comments(model, filepath, dataset_name):
    print(f"\nPredicting on dataset: {filepath}")
    df = pd.read_csv(filepath)
    
    # Preprocess
    df['CONTENT'] = df['CONTENT'].apply(clean_text)
    
    # Extract features
    features = df['CONTENT'].apply(extract_features)
    for i, name in enumerate(feature_names):
        df[name] = [f[i] for f in features]
    
    dataset = CommentDataset(df, vocab, MAX_LEN, feature_names)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            text = batch['text'].to(device)
            features = batch['features'].to(device) if batch['features'] is not None else None
            
            outputs = model(text, features)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of being a bot

    total_comments = len(all_preds)
    bot_comments = sum(all_preds)
    human_comments = total_comments - bot_comments
    
    # Calculate confidence levels for bot predictions
    high_confidence_bots = sum(1 for pred, prob in zip(all_preds, all_probs) if pred == 1 and prob > 0.8)
    medium_confidence_bots = sum(1 for pred, prob in zip(all_preds, all_probs) if pred == 1 and 0.6 <= prob <= 0.8)
    low_confidence_bots = sum(1 for pred, prob in zip(all_preds, all_probs) if pred == 1 and prob < 0.6)
    
    # Store results for plotting
    prediction_results[dataset_name] = {
        'Bot Comments': bot_comments,
        'Human Comments': human_comments,
        'Total': total_comments,
        'Bot Confidence': {
            'High': high_confidence_bots,
            'Medium': medium_confidence_bots,
            'Low': low_confidence_bots
        }
    }
    
    print(f"Total comments: {total_comments}")
    print(f"Detected bot comments: {bot_comments} ({100*bot_comments/total_comments:.2f}%)")
    print(f"Detected human comments: {human_comments} ({100*human_comments/total_comments:.2f}%)")
    print(f"Bot confidence - High: {high_confidence_bots}, Medium: {medium_confidence_bots}, Low: {low_confidence_bots}")

# Predict on the datasets

hobbits_file = 'data/Real Comments/hobbits_to_isengard.csv'
predict_bot_comments(model, hobbits_file, 'Hobbits To Isengard')

# Create visualization
plt.figure(figsize=(15, 10))

# 1. Plot accuracy over epochs
plt.figure(figsize=(8, 6))
plt.plot(epoch_numbers, test_accuracy_history, marker='o', linestyle='-', color='blue')
plt.title('Test Accuracy over Epochs', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Test Accuracy (%)', fontsize=12)
plt.ylim(0, 100)
plt.grid(True)
plt.show()

# 2-4. Pie charts for each dataset
datasets = list(prediction_results.keys())
positions = [2, 3, 4]  # Subplot positions

for i, dataset_name in enumerate(datasets):
    plt.subplot(2, 2, positions[i])
    
    data = prediction_results[dataset_name]
    labels = ['Bot Comments', 'Human Comments']
    sizes = [data['Bot Comments'], data['Human Comments']]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.1, 0)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.axis('equal')
    plt.title(f'Comments: {dataset_name}', fontsize=14)
    
    # Add confidence breakdown for bots
    bot_confidence = data['Bot Confidence']
    confidence_text = f"Total: {data['Total']}\nBot Confidence:\n"
    
    # Only calculate percentages if there are bot comments
    if data['Bot Comments'] > 0:
        confidence_text += f"High: {bot_confidence['High']} ({100*bot_confidence['High']/data['Bot Comments']:.1f}%)\n"
        confidence_text += f"Medium: {bot_confidence['Medium']} ({100*bot_confidence['Medium']/data['Bot Comments']:.1f}%)\n" 
        confidence_text += f"Low: {bot_confidence['Low']} ({100*bot_confidence['Low']/data['Bot Comments']:.1f}%)"
    else:
        confidence_text += "No bots detected"
    
    plt.annotate(confidence_text, 
                 xy=(0.5, 0.01),
                 xycoords='axes fraction',
                 ha='center',
                 va='bottom',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.savefig('bot_detection_results.png', dpi=300)
plt.show()


with open("rnn_metrics", "w", encoding="utf-8") as file:
    file.write("precision: %.4f\n" % precision_per_epoch[best_epoch])
    file.write("recall: %.4f\n" % recall_per_epoch[best_epoch])
    file.write("accuracy: %.4f\n" % accuracy_history[best_epoch])
    file.write("f1_score: %.4f\n" % f1_per_epoch[best_epoch])
    file.write("auc: %.4f\n" % auc_per_epoch[best_epoch])
