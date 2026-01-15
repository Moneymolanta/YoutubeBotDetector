import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(torch.nn.Module):
    """
    CNN model to test.
    """

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
        batch_size = output.size(0)
        output = output.view(batch_size, -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output