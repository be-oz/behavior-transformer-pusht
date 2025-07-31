# fast_prototype_pushT.py
# One-file setup: dataset generation + training + evaluation

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random, os

# --- Config ---
D_INPUT = 5
D_MODEL = 64
NHEAD = 2
NUM_LAYERS = 1
NUM_BINS = 32
BATCH_SIZE = 256
NUM_EPOCHS = 3
LR = 1e-4
DATASET_SIZE = 2000
MODEL_PATH = "my_policy.pt"


# --- Synthetic Dataset Generator ---
def generate_synthetic_dataset(n_episodes=DATASET_SIZE):
    data = []
    for _ in range(n_episodes):
        T = random.randint(10, 50)
        obs = np.random.uniform(-1, 1, size=(T, D_INPUT)).astype(np.float32)
        action_bin = np.random.randint(0, NUM_BINS, size=(T,))
        residual = np.random.normal(0, 0.05, size=(T, 4)).astype(np.float32)
        data.append({"obs": obs, "action_bin": action_bin, "residual": residual})
    np.save("transformer_dataset.npy", data)
    print(f"Generated {len(data)} episodes of synthetic data.")


# --- Dataset ---
class PushTDataset(Dataset):
    def __init__(self, path="transformer_dataset.npy"):
        data = np.load(path, allow_pickle=True)
        self.obs, self.bins, self.res = [], [], []
        for ep in data:
            self.obs.append(ep["obs"])
            self.bins.append(ep["action_bin"])
            self.res.append(ep["residual"])
        self.obs = np.concatenate(self.obs)
        self.bins = np.concatenate(self.bins)
        self.res = np.concatenate(self.res)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, i):
        return (
            torch.tensor(self.obs[i], dtype=torch.float32),
            torch.tensor(self.bins[i], dtype=torch.long),
            torch.tensor(self.res[i], dtype=torch.float32),
        )


# --- Model ---
class FastTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(D_INPUT, D_MODEL)
        encoder = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=NHEAD, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=NUM_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.cls_head = nn.Linear(D_MODEL, NUM_BINS)
        self.reg_head = nn.Linear(D_MODEL, 4)

    def forward(self, x):
        x = self.embed(x.unsqueeze(1))
        x = self.transformer(x)
        x = self.norm(x.squeeze(1))
        return self.cls_head(x), self.reg_head(x)


# --- Training ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🖥️ Device:", device)
    dataset = PushTDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = FastTransformer().to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_cls = nn.CrossEntropyLoss()
    loss_reg = nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        cls_total, reg_total = 0, 0
        for x, b, r in tqdm(loader, desc=f"Epoch {epoch+1}"):
            x, b, r = x.to(device), b.to(device), r.to(device)
            out_cls, out_reg = model(x)
            loss = loss_cls(out_cls, b) + loss_reg(out_reg, r)
            opt.zero_grad(); loss.backward(); opt.step()
            cls_total += loss_cls(out_cls, b).item() * len(x)
            reg_total += loss_reg(out_reg, r).item() * len(x)
        print(f"Epoch {epoch+1} | Cls Loss: {cls_total/len(dataset):.4f} | Reg Loss: {reg_total/len(dataset):.4f}")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")



if __name__ == "__main__":
    generate_synthetic_dataset()
    train()
