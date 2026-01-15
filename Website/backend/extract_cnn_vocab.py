import torch
import pickle

ckpt_path = "best_CNN_model.pth"
output_vocab_path = "cnn_vocab.pkl"

checkpoint = torch.load(ckpt_path, map_location="cpu")
print("Checkpoint keys:", checkpoint.keys())

# Try common vocab keys
possible_vocab_keys = ["vocab", "word2idx", "token2idx"]
vocab = None
for key in possible_vocab_keys:
    if key in checkpoint:
        vocab = checkpoint[key]
        print(f"Found vocab under key: {key}")
        break

if vocab is not None:
    with open(output_vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary extracted and saved to {output_vocab_path}")
else:
    print("No vocabulary found in checkpoint. You may need to retrain and save the vocab.")
