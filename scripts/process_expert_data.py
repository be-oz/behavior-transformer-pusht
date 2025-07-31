import numpy as np
import torch
from sklearn.cluster import KMeans
import os

# --- Config ---
N_CLUSTERS = 32
INPUT_PATH = "expert_data/expert_trajectories.npy"
OUTPUT_DATASET = "transformer_dataset.npy"
CODEBOOK_PATH = "action_codebook.pt"

# --- Load expert trajectories ---
expert_episodes = np.load(INPUT_PATH, allow_pickle=True)

all_actions = []
all_obs = []
bin_indices = []
residuals = []

# --- Aggregate actions for KMeans ---
for episode in expert_episodes:
    for step in episode:
        all_obs.append(step["obs"])
        all_actions.append(step["action"])

all_actions = np.array(all_actions, dtype=np.float64)
all_obs = np.array(all_obs)

# --- Fit KMeans codebook ---
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42).fit(all_actions)
codebook = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
torch.save(codebook, CODEBOOK_PATH)
print(f"✅ Saved KMeans action codebook to {CODEBOOK_PATH}")

# --- Build training dataset with bins + residuals ---
dataset = []
i = 0
for episode in expert_episodes:
    obs_seq, bin_seq, res_seq = [], [], []
    for step in episode:
        obs = step["obs"]
        act = step["action"]
        bin_idx = kmeans.predict([act.astype(np.float64)])[0]
        res = act - codebook[bin_idx].numpy()
        obs_seq.append(obs)
        bin_seq.append(bin_idx)
        res_seq.append(res)
        i += 1
    dataset.append({
        "obs": np.array(obs_seq, dtype=np.float32),
        "action_bin": np.array(bin_seq, dtype=np.int64),
        "residual": np.array(res_seq, dtype=np.float32)
    })

np.save(OUTPUT_DATASET, dataset)
print(f"✅ Saved processed dataset to {OUTPUT_DATASET} with {i} steps.")
