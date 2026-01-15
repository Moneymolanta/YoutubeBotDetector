from bert import BERTWithMetadata
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.preprocessing import StandardScaler
from transformers import DistilBertTokenizer

import matplotlib.pyplot as plt 
# Load the dataset
df = pd.read_csv('data/Youtube Spam Dataset/Youtube04-Eminem.csv')

# Define a function to clean the text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = text.strip()
    return text

df['CONTENT'] = df['CONTENT'].apply(clean_text)

# Check for NaN values in columns
# nan_check = df.isna().sum()
# print("NaN values in columns:", nan_check)

df = df.dropna(subset=['DATE']).copy()

# Process metadata features
df['DATE'] = pd.to_datetime(df['DATE'], format='ISO8601')
df['content_length'] = df['CONTENT'].apply(len)
df['day_of_week'] = df['DATE'].dt.dayofweek
df['hour'] = df['DATE'].dt.hour

# # Normalize metadata features
scaler = StandardScaler()
metadata_features = scaler.fit_transform(df[['content_length', 'day_of_week', 'hour']].values)
meta_tensor = torch.tensor(metadata_features, dtype=torch.float)

# Check raw metadata before scaling
print("Raw metadata NaN check:", df[['content_length', 'day_of_week', 'hour']].isna().sum())

# Check scaled metadata
print("Scaled metadata NaN check:", np.isnan(metadata_features).sum())

# Verify StandardScaler
print("Scaler mean:", scaler.mean_)
print("Scaler scale:", scaler.scale_)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
encodings = tokenizer(list(df['CONTENT']), padding=True, truncation=True, return_tensors='pt')

max_length = 128

class CommentDataset(Dataset):
    def __init__(self, texts, metadata, labels, tokenizer, max_length):
        self.texts = texts
        self.metadata = metadata
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        metadata = self.metadata[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'metadata': torch.tensor(metadata, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.float)
        }
        

# Split data
train_texts, temp_texts, train_metadata, temp_metadata, train_labels, temp_labels = train_test_split(
    df['CONTENT'].tolist(),
    metadata_features,
    df['CLASS'].values,
    test_size=0.2,
    random_state=42
)

val_texts, test_texts, val_metadata, test_metadata, val_labels, test_labels = train_test_split(
    temp_texts,
    temp_metadata,
    temp_labels,
    test_size=0.5,
    random_state=42
)

# Create datasets
train_dataset = CommentDataset(train_texts, train_metadata, train_labels, tokenizer, max_length)
# Test single sample
# sample = train_dataset[0]
# print("\n=== Sample Inspection ===")
# print(f"Input IDs shape: {sample['input_ids'].shape}")
# print(f"Attention mask shape: {sample['attention_mask'].shape}")
# print(f"Metadata shape: {sample['metadata'].shape}")
# print(f"Label value: {sample['label']}")
# print("Input IDs:", sample['input_ids'])
# print("Metadata:", sample['metadata'])
# print("Label:", sample['label'])

val_dataset = CommentDataset(val_texts, val_metadata, val_labels, tokenizer, max_length)
test_dataset = CommentDataset(test_texts, test_metadata, test_labels, tokenizer, max_length)

# Create DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# Check first batch
# first_batch = next(iter(train_loader))
# print("\n=== Batch Inspection ===")
# print(f"Batch size: {len(first_batch['label'])}")
# print("Input IDs shape:", first_batch['input_ids'].shape)
# print("Attention mask shape:", first_batch['attention_mask'].shape)
# print("Metadata shape:", first_batch['metadata'].shape)
# print("Labels:", first_batch['label'])

# # Check value ranges
# print("\nValue Ranges:")
# print(f"Input IDs min/max: {first_batch['input_ids'].min()} - {first_batch['input_ids'].max()}")
# print(f"Metadata min/max: {first_batch['metadata'].min()} - {first_batch['metadata'].max()}")
# print(f"Labels unique: {torch.unique(first_batch['label'])}")

val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

mps_enabled = torch.backends.mps.is_available() and torch.backends.mps.is_built()

device = torch.device('mps' if mps_enabled else 'cpu')
model = BERTWithMetadata(metadata_dim=3, dropout=0.3).to(device)


# Print parameters with names and shapes
def print_model_parameters(model):
    print("=" * 60)
    print("Model Parameters:")
    print("=" * 60)
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            params = param.numel()
            total_params += params
            print(f"{name:<60} | Shape: {str(param.shape):<20} | Parameters: {params:,}")
    print("=" * 60)
    print(f"Total Trainable Parameters: {total_params:,}")
    print("=" * 60)

# Run the parameter printer
# print_model_parameters(model)

# Optional: Print parameter values (use with caution!)
# for name, param in model.named_parameters():
#     print(f"\n{name}:")
#     print(param.data)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5,
                            weight_decay=0.01)
criterion = nn.BCEWithLogitsLoss()

num_epochs = 3

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    # Training
    model.train()
    total_loss = 0
    correct_train = 0  
    total_train = 0  
    
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        metadata = batch['metadata'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids, attention_mask, metadata)
        # print(f"Output range: {outputs.min().item():.4f} - {outputs.max().item():.4f}")
        loss = criterion(outputs.view(-1), labels)
        loss.backward()
        optimizer.step()
        
        preds = (outputs > 0.5).float()
        correct_train += (preds.view(-1) == labels).sum().item()
        total_train += labels.size(0)
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}')
    
    train_acc = correct_train / total_train
    
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc * 100)

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            metadata = batch['metadata'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, metadata)
            # print(f"Output range: {outputs.min().item():.4f} - {outputs.max().item():.4f}")
            val_loss += criterion(outputs.view(-1), labels).item()
            
            preds = (outputs > 0.5).float()
            correct += (preds.view(-1) == labels).sum().item()
            total += labels.size(0)
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}\n')
    
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc * 100)
    

model.eval()
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        metadata = batch['metadata'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids, attention_mask, metadata)
        test_loss += criterion(outputs.view(-1), labels).item()
        
        preds = (outputs > 0.5).float()
        correct += (preds.view(-1) == labels).sum().item()
        total += labels.size(0)

avg_test_loss = test_loss / len(test_loader)
test_acc = correct / total
print(f'Test Loss: {avg_test_loss:.4f}, Accuracy: {test_acc:.4f}')

# # Plot training and validation accuracy
# plt.figure(figsize=(10, 5))
# epochs_range = range(1, num_epochs + 1)

# plt.plot(epochs_range, train_accuracies, 'bo-', label='Training Accuracy')
# plt.plot(epochs_range, val_accuracies, 'ro-', label='Validation Accuracy')
# plt.title('Accuracy per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.xticks(epochs_range)
# plt.legend()
# plt.grid(linestyle='--')
# plt.show()

plt.figure(figsize=(10, 5))
epochs_range = range(1, num_epochs + 1)

# Simple line graph
plt.plot(epochs_range, train_accuracies, 
         color='blue',  
         linewidth=2,   
         linestyle='-', 
         label='Training Accuracy')

plt.title('Training Accuracy per Epoch', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(epochs_range)
plt.ylim(0, 100) 
plt.legend(loc='lower right')
plt.grid(linestyle='--', alpha=0.6)
plt.show()

plt.savefig("training_accuracy_per_epoch.png")
print("Training accuracy plot saved as training_accuracy_per_epoch.png.")

