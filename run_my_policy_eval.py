# run_my_policy_eval.py
"""
Evaluates a trained policy on the PushT-v0 environment and records a rollout video.
This script uses a Transformer-based policy to predict actions based solely on the 2D agent position.
"""

import os, sys
import torch
import gymnasium as gym
import gym_pusht
import numpy as np
import imageio
from pathlib import Path

# --- Adding local modules to import path ---
sys.path.append(os.path.dirname(__file__))

from my_policy import MyPolicy

# -------------------- Configuration --------------------
ENV_ID = "gym_pusht/PushT-v0"
OBS_TYPE = "pixels_agent_pos"  # includes both image and 2D position
MAX_STEPS = 300
SEED = 42
VIDEO_PATH = "outputs/my_policy_eval.mp4"
RENDER_FPS = 30

# -------------------- Device + Policy Setup --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = MyPolicy(device=device)

# Initialize environment
env = gym.make(ENV_ID, obs_type=OBS_TYPE, max_episode_steps=MAX_STEPS)
obs, _ = env.reset(seed=SEED)

# -------------------- Rollout --------------------
frames = [env.render()]
rewards = []
done = False

while not done:
    # Extract 2D agent state (required by policy)
    state = torch.tensor(obs["agent_pos"][:2], dtype=torch.float32, device=device).unsqueeze(0)  # shape: (1, 2)
    
    # Image input (not used by current policy, but included for future extensibility)
    image = torch.tensor(obs["pixels"], dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0

    # Compute action
    action = policy.act({
        "observation.state": state,
        "observation.image": image  # Ignored in current implementation
    }).squeeze(0).cpu().numpy()

    # Step environment
    obs, reward, terminated, truncated, _ = env.step(action)
    rewards.append(reward)
    frames.append(env.render())
    done = terminated or truncated

# -------------------- Output --------------------
total_reward = sum(rewards)
print("Evaluation finished.")
print(f"Total reward: {total_reward:.4f}")

# Save video
Path(os.path.dirname(VIDEO_PATH)).mkdir(parents=True, exist_ok=True)
imageio.mimsave(VIDEO_PATH, np.stack(frames), fps=RENDER_FPS)
print(f"Saved rollout to {VIDEO_PATH}")
