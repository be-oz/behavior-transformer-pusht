# train_transformer.py
"""
Train a lightweight Transformer for behavior cloning on the PushT-v0 task
using 2D agent position as input and expert actions as target outputs.

This model predicts a coarse action bin (via classification) and a fine-grained
residual vector (via regression). The hybrid formulation allows us to discretize
the action space for stability while preserving precision with residual learning.

Author: Berke O. / https://www.linkedin.com/in/mars666/ - 2025
"""

import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from tqdm import tqdm

# --- Static Configurations ---
D_INPUT = 2           # 2D agent position input (x, y)
D_MODEL = 64          # Transformer hidden dimension
NHEAD = 2             # Number of attention heads
NUM_LAYERS = 1        # Single encoder layer (lightweight)
NUM_BINS = 32         # Discretized action bins (KMeans clusters)
LR = 1e-4             # Learning rate
BATCH_SIZE = 256
NUM_EPOCHS = 10
DEVICE = "cpu"

# --- File Paths ---
DATA_PATH = os.path.join("expert_data", "expert_dataset.npy")
CODEBOOK_PATH = os.path.join("expert_data", "action_codebook.pt")
MODEL_PATH = "my_policy.pt"

# ----------------------------
# Dataset: Loads expert demos
# ----------------------------
class PushTDataset(Dataset):
    def __init__(self):
        data = np.load(DATA_PATH, allow_pickle=True)
        self.obs = np.concatenate([ep["obs"] for ep in data])          # shape: (N, 2)
        self.res = np.concatenate([ep["action"] for ep in data])       # shape: (N, 2)
        self.bins = np.zeros(len(self.res), dtype=np.int64)            # placeholder, to be set via KMeans

    def __len__(self): return len(self.obs)

    def __getitem__(self, i):
        return (
            torch.tensor(self.obs[i], dtype=torch.float32),
            torch.tensor(self.bins[i], dtype=torch.long),
            torch.tensor(self.res[i], dtype=torch.float32),
        )

# ----------------------------
# Transformer Model
# ----------------------------
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(D_INPUT, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=NHEAD, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.cls_head = nn.Linear(D_MODEL, NUM_BINS)       # Predicts discrete bin index
        self.reg_head = nn.Linear(D_MODEL, D_INPUT)        # Predicts residual offset

    def forward(self, x):
        x = self.embed(x.unsqueeze(1))                     # (B, 1, D_MODEL)
        x = self.transformer(x)                            # (B, 1, D_MODEL)
        x = self.norm(x.squeeze(1))                        # (B, D_MODEL)
        return self.cls_head(x), self.reg_head(x)          # (B, NUM_BINS), (B, 2)

# ----------------------------
# Training Loop
# ----------------------------
def train():
    dataset = PushTDataset()

    # Step 1: Build action codebook via KMeans
    kmeans = KMeans(n_clusters=NUM_BINS, random_state=42, n_init=10)
    kmeans.fit(dataset.res)
    dataset.bins = kmeans.predict(dataset.res)
    codebook = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    torch.save(codebook, CODEBOOK_PATH)

    # Step 2: Model training
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = TransformerModel().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_cls = nn.CrossEntropyLoss()
    loss_reg = nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        cls_total, reg_total = 0.0, 0.0

        for x, bin_idx, residual in tqdm(loader, desc=f"Epoch {epoch + 1}"):
            x, bin_idx, residual = x.to(DEVICE), bin_idx.to(DEVICE), residual.to(DEVICE)

            logits, pred_residual = model(x)
            loss = loss_cls(logits, bin_idx) + loss_reg(pred_residual, residual)

            opt.zero_grad()
            loss.backward()
            opt.step()

            cls_total += loss_cls(logits, bin_idx).item() * len(x)
            reg_total += loss_reg(pred_residual, residual).item() * len(x)

        print(f"Epoch {epoch + 1}: Cls Loss = {cls_total / len(dataset):.4f}, Reg Loss = {reg_total / len(dataset):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
