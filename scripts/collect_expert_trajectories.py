# collect_and_bin_expert_data.py
"""
Collect expert demonstrations from the PushT-v0 environment and process them
into a behavior cloning dataset by computing discrete action bins (via KMeans)
and fine-grained residuals.

Author: Berke Özmen
"""

import os
import numpy as np
import torch
import gymnasium as gym
import gym_pusht
import lerobot.envs.pusht
from sklearn.cluster import KMeans

# ----------------------------- Configuration ----------------------------- #
ENV_NAME = "gym_pusht/PushT-v0"
OBS_TYPE = "pixels_agent_pos"
MAX_EPISODE_STEPS = 300
NUM_EPISODES = 500
NUM_BINS = 32
SEED = 42
D_INPUT = 2

SAVE_DIR = "expert_data"
SAVE_PATH = os.path.join(SAVE_DIR, "expert_dataset.npz")
CODEBOOK_PATH = os.path.join(SAVE_DIR, "action_codebook.pt")

# ----------------------------- Initialization ----------------------------- #
env = gym.make(ENV_NAME, obs_type=OBS_TYPE, max_episode_steps=MAX_EPISODE_STEPS)
env.reset(seed=SEED)
rng = np.random.default_rng(SEED)

os.makedirs(SAVE_DIR, exist_ok=True)

obs_list = []
action_list = []

# ----------------------------- Expert Collection ----------------------------- #
print(f"Collecting {NUM_EPISODES} expert trajectories from {ENV_NAME}...")

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False

    while not done:
        agent_pos = obs["agent_pos"][:D_INPUT]  # Only x, y position
        expert_action = env.unwrapped.get_expert_action()

        obs_list.append(agent_pos)
        action_list.append(expert_action)

        obs, _, terminated, truncated, _ = env.step(expert_action)
        done = terminated or truncated

    print(f"Episode {ep + 1}/{NUM_EPISODES} collected")

obs_array = np.array(obs_list, dtype=np.float32)          # shape: (N, 2)
action_array = np.array(action_list, dtype=np.float32)    # shape: (N, 2)

# ----------------------------- KMeans Binning ----------------------------- #
print(f"\nFitting KMeans with {NUM_BINS} clusters for coarse action binning...")

kmeans = KMeans(n_clusters=NUM_BINS, random_state=SEED, n_init=10)
bin_labels = kmeans.fit_predict(action_array)
centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)  # (NUM_BINS, 2)

# Compute fine-grained residuals
residuals = action_array - centroids[bin_labels].numpy()  # shape: (N, 2)

# ----------------------------- Save Dataset ----------------------------- #
np.savez(SAVE_PATH, obs=obs_array, bin_idx=bin_labels, residual=residuals)
torch.save(centroids, CODEBOOK_PATH)

print(f"\nSaved expert dataset to: {SAVE_PATH}")
print(f"Saved action codebook to: {CODEBOOK_PATH}")
