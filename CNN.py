import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
import re
import pandas as pd
from collections import Counter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('data/Youtube Spam Dataset/Youtube-Spam-Dataset.csv')
real_comments_df = pd.read_csv('validation_data.csv')

# Define a function to clean the text
def clean_text(text):
    # text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = text.strip()
    return text

# Clean the text data
df['CONTENT'] = df['CONTENT'].apply(clean_text)
real_comments_df['CONTENT'] = real_comments_df['CONTENT'].apply(clean_text)

# Tokenize the text
def tokenize_text(text):
    return text.split()

# Create a vocabulary
all_tokens = []
for text in df['CONTENT']:
    tokens = tokenize_text(text)
    all_tokens.extend(tokens)

# Create a vocabulary
real_comments_all_tokens = []
for text in real_comments_df['CONTENT']:
    tokens = tokenize_text(text)
    real_comments_all_tokens.extend(tokens)

vocab = Counter(all_tokens)
vocab = {word: idx for idx, (word, _) in enumerate(vocab.most_common())}

real_comments_vocab = Counter(real_comments_all_tokens)
real_comments_vocab = {word: idx for idx, (word, _) in enumerate(real_comments_vocab.most_common())}

# Create a dataset class
class CommentDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.df = df
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx, 3]
        label = self.df.iloc[idx, 5]

        # Tokenize and convert to tensor
        tokens = tokenize_text(text)
        tokens = [self.vocab.get(token, 0) for token in tokens]  # Use 0 for unknown tokens
        tokens = tokens[:self.max_len]  # Truncate if necessary
        tokens += [0] * (self.max_len - len(tokens))  # Pad if necessary

        return {
            'text': torch.tensor(tokens),
            'label': torch.tensor(label, dtype=torch.long)
        }

class ValDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.df = df
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx, 3]

        # Tokenize and convert to tensor
        tokens = tokenize_text(text)
        tokens = [self.vocab.get(token, 0) for token in tokens]  # Use 0 for unknown tokens
        tokens = tokens[:self.max_len]  # Truncate if necessary
        tokens += [0] * (self.max_len - len(tokens))  # Pad if necessary

        return {
            'text': torch.tensor(tokens),
        }

# Create dataset and data loader
dataset = CommentDataset(df, vocab, max_len=100)
real_comments_dataset = ValDataset(real_comments_df, vocab, max_len=100)
batch_size = 32

# Split datasets
dataset_len = len(dataset)
train_size = int(0.7 * dataset_len)
test_size = dataset_len - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loader
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
real_comments_data_loader = DataLoader(real_comments_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


class CNN(torch.nn.Module):
    """
    CNN model to test.
    """

    def __init__(self, batch_size, embedding_dim, output_dim, vocab_size):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, batch_size, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv1d(batch_size, 4, 5)
        self.fc1 = nn.Linear(368, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output = F.relu(self.conv1(embedded))
        # output = F.relu(self.pool(output))
        output = F.sigmoid(self.conv2(output))
        # output = F.sigmoid(self.pool(output))
        output = output.view(32, -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output


if __name__ == "__main__":

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")    # Mac GPU
    elif torch.cuda.is_available():
        device = torch.device("cuda")   # NVIDIA GPU
    else:
        device = torch.device("cpu")    # CPU fallback
    print(f"Using device: {device}")

    # Initialize the model, optimizer, and loss function
    vocab_size = len(vocab) + 1  # +1 for padding token
    model = CNN(batch_size=32, embedding_dim=100, output_dim=2, vocab_size=vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    print(f'vocab_size {vocab_size}')

    # Train the model
    model.to(device)
    accuracy_per_epoch = []
    for epoch in range(25):  # Train for 5 epochs
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        test_count = 0
        for batch in train_data_loader:
            # print(f'batch {test_count}: {len(batch)}, {batch_size}')
            test_count += 1
            text = batch['text'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            outputs = model(text)
            # print(f'outputs shape: {outputs.shape}')
            # print(f'labels shape: {labels.shape}')
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        accuracy_per_epoch.append(100 * accuracy)
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_data_loader)}, Accuracy: {accuracy:.4f}')

    # Evaluate the model
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch in test_data_loader:
            text = batch['text'].to(device)
            labels = batch['label'].to(device)

            outputs = model(text)
            _, predicted = torch.max(outputs, dim=1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

    test_accuracy = test_correct / test_total
    print(f'Test Accuracy: {test_accuracy:.4f}')




    # Try the real comments dataset.
    real_correct = 0
    real_total = 0
    predicted_labels = []
    with torch.no_grad():
        for batch in real_comments_data_loader:
            text = batch['text'].to(device)
            # labels = batch['label'].to(device)

            outputs = model(text)

            _, predicted = torch.max(outputs, dim=1)
            predicted_labels = predicted_labels + predicted.tolist()
            # print(predicted)
            real_correct += sum(predicted)
            real_total += len(predicted)

    bot_percent = real_correct / real_total
    print(f'Bot Percent: {bot_percent:.4f}')

    with open("predicted_labels.csv", "w", encoding="utf-8") as file:
        for label in predicted_labels:
            file.write(f"{label}\n")

    # Create plot
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.ylim((0, 100))
    plt.xlim((0,25))
    plt.plot([i for i in range(len(accuracy_per_epoch))], accuracy_per_epoch)
    plt.show()
