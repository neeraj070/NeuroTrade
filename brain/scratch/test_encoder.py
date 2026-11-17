import torch
import torch.nn as nn
import numpy as np

# Load encoder
ckpt = torch.load("backend/checkpoints/improved_rnn_encoder.pt", weights_only=True)

class MarketStateLSTMEncoder(nn.Module):
    def __init__(self, input_dim=14, sequence_length=30, hidden_dim=64, num_layers=2,
                 embedding_dim=32, output_dim=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                           num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.embedding_layer = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.LeakyReLU(negative_slope=0.1)
        )
        self.regressor = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(embedding_dim, output_dim))

    def encode(self, x):
        out, _ = self.lstm(x)
        return self.embedding_layer(out[:, -1, :])

model = MarketStateLSTMEncoder(**ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

for split, xp, ep in [
    ("train", "data/training/train/X_train.pt", "backend/embeddings/train_embeddings_improved.pt"),
    ("val", "data/training/validation/X_val.pt", "backend/embeddings/val_embeddings_improved.pt"),
    ("test", "data/training/test/X_test.pt", "backend/embeddings/test_embeddings_improved.pt")
]:
    x = torch.load(xp, weights_only=True)
    saved_emb = torch.load(ep, weights_only=True)
    with torch.no_grad():
        computed_emb = model.encode(x)
    max_diff = (computed_emb - saved_emb).abs().max().item()
    print(f"{split}: max diff between model.encode(X) and saved embeddings = {max_diff:.8e}")
    assert max_diff < 1e-5, f"Mismatch in {split}!"
print("Encoder model.encode() matches saved embeddings exactly [VERIFIED!]")

