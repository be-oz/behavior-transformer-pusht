# mlp_bc_train.py
# Author: Berke Özmen
# Description: Clean MLP behavior cloning pipeline with verified dataset structure from PushT (LeRobot)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ===============================
# CONFIGURATION
# ===============================
USE_RESIDUAL = True         # Enable regression head for hybrid prediction
KMEANS_BIN_COUNT = 32       # Number of discrete bins for coarse action classification
EPOCHS = 10
BATCH_SIZE = 256
LR = 1e-3

# ===============================
# MLP MODEL DEFINITION
# ===============================
class MLPPolicy(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, n_bins=32, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim, n_bins)
        if use_residual:
            self.regressor = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        h = self.encoder(x)
        cls_out = self.classifier(h)
        reg_out = self.regressor(h) if self.use_residual else None
        return cls_out, reg_out

# ===============================
# LOAD AND TRANSFORM DATASET (verified structure)
# ===============================
raw = np.load("transformer_dataset.npy", allow_pickle=True)

# Each entry is a list or tuple: [obs, bin_idx, residual]
obs = np.stack([entry[0] for entry in raw])         # shape: [N, 2]
bin_idx = np.array([entry[1] for entry in raw])     # shape: [N]
residual = np.stack([entry[2] for entry in raw])    # shape: [N, 2]

X = torch.tensor(obs, dtype=torch.float32)
y_cls = torch.tensor(bin_idx, dtype=torch.long)
y_reg = torch.tensor(residual, dtype=torch.float32)

# ===============================
# SPLIT + DATALOADER
# ===============================
X_train, X_val, y_cls_train, y_cls_val, y_reg_train, y_reg_val = train_test_split(
    X, y_cls, y_reg, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train, y_cls_train, y_reg_train)
val_dataset = TensorDataset(X_val, y_cls_val, y_reg_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ===============================
# TRAINING SETUP
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPPolicy(n_bins=KMEANS_BIN_COUNT, use_residual=USE_RESIDUAL).to(device)
criterion_cls = nn.CrossEntropyLoss()
criterion_reg = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

cls_losses, reg_losses = [], []

# ===============================
# TRAINING LOOP
# ===============================
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_cls_loss = 0.0
    total_reg_loss = 0.0

    for xb, yb_cls, yb_reg in train_loader:
        xb, yb_cls, yb_reg = xb.to(device), yb_cls.to(device), yb_reg.to(device)

        pred_cls, pred_reg = model(xb)
        loss_cls = criterion_cls(pred_cls, yb_cls)
        loss = loss_cls

        if USE_RESIDUAL:
            loss_reg = criterion_reg(pred_reg, yb_reg)
            loss += loss_reg
            total_reg_loss += loss_reg.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_cls_loss += loss_cls.item()

    avg_cls = total_cls_loss / len(train_loader)
    cls_losses.append(avg_cls)

    if USE_RESIDUAL:
        avg_reg = total_reg_loss / len(train_loader)
        reg_losses.append(avg_reg)

    print(f"Epoch {epoch+1:02d} | ClsLoss: {avg_cls:.4f} | RegLoss: {avg_reg if USE_RESIDUAL else 'N/A'}")

# ===============================
# PLOTTING LOSS CURVES
# ===============================
plt.figure(figsize=(10, 5))
plt.plot(cls_losses, label="Classification Loss")
if USE_RESIDUAL:
    plt.plot(reg_losses, label="Regression Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curves (MLP)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mlp_bc_loss.png")
plt.show()

# ===============================
# SAVE TRAINED MODEL
# ===============================
filename = f"mlp_policy_k{KMEANS_BIN_COUNT}_residual{USE_RESIDUAL}.pt"
torch.save(model.state_dict(), filename)
print(f"Model saved to: {filename}")