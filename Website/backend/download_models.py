#!/usr/bin/env python

print("Starting model download...")

import os
import torch
from transformers import DistilBertTokenizer, DistilBertModel

# Create model directory
os.makedirs("model_files", exist_ok=True)

print("Downloading DistilBERT tokenizer...")
# Download and save the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer.save_pretrained('./model_files')

print("Downloading DistilBERT model...")
# Download and save the model
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model.save_pretrained('./model_files')

print("Download complete! Files saved to ./model_files")
print("Model files:")
for file in os.listdir("./model_files"):
    file_path = os.path.join("./model_files", file)
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"  - {file} ({size_mb:.2f} MB)")
