import os
import numpy as np
import torch
import gymnasium as gym
import gym_pusht
from tqdm import trange

# Configuration
ENV_NAME = "gym_pusht/PushT-v0"
OBS_TYPE = "pixels_agent_pos"
RENDER_MODE = "rgb_array"
NUM_EPISODES = 100
MAX_STEPS = 300
SAVE_PATH = "expert_data/expert_trajectories.npy"

# Setup
env = gym.make(ENV_NAME, obs_type=OBS_TYPE, max_episode_steps=MAX_STEPS, render_mode=RENDER_MODE)

# ---- Mock Expert Policy ----
def expert_policy(obs: dict) -> np.ndarray:
    """
    Computes a goal-directed action using the agent position.
    This is a placeholder for a learned or classical control policy.
    """
    agent_pos = np.array(obs["agent_pos"])  # (2,)
    goal = np.array([0.75, 0.0])  # Hardcoded target for demonstration
    direction = goal - agent_pos
    action = direction / (np.linalg.norm(direction) + 1e-8)  # Normalize
    action = np.clip(action * 0.1, -1.0, 1.0)  # Scaled step
    return action.astype(np.float32)

# ---- Data Collection ----
expert_trajectories = []

for ep in trange(NUM_EPISODES, desc="Collecting expert trajectories"):
    obs, _ = env.reset(seed=ep)
    episode = []

    for _ in range(MAX_STEPS):
        action = expert_policy(obs)
        episode.append({
            "obs": obs["agent_pos"],   # 2D position
            "action": action           # 2D action
        })
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    expert_trajectories.append(episode)

# ---- Save ----
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
np.save(SAVE_PATH, expert_trajectories)
print(f"Saved {NUM_EPISODES} expert episodes to {SAVE_PATH}")
