# run_mlp_policy_eval.py
# Author: Berke Özmen
# Description: Evaluate MLP policy on PushT, save video and visualize agent trace

import gym
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
from mlp_policy import MLPPolicy

# ===============================
# CONFIGURATION
# ===============================
MODEL_PATH = "mlp_policy_k32_residualTrue.pt"
CODEBOOK_PATH = "action_codebook.pt"
USE_RESIDUAL = True
EPISODE_STEPS = 300
RENDER_FPS = 15
SAVE_VIDEO = True

# ===============================
# ENV SETUP
# ===============================
env = gym.make("PushT-v0", render_mode="rgb_array")
obs, _ = env.reset()

# ===============================
# LOAD POLICY AND CODEBOOK
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPPolicy(use_residual=USE_RESIDUAL).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

codebook = torch.load(CODEBOOK_PATH)

# ===============================
# ROLLOUT + VIDEO COLLECTION
# ===============================
frames = []
agents = []
rewards = []

for _ in range(EPISODE_STEPS):
    frame = env.render()
    frames.append(frame)
    agents.append(obs[:2])

    obs_tensor = torch.tensor(obs[:2], dtype=torch.float32).to(device)
    action = model.act(obs_tensor, codebook).detach().cpu().numpy()

    obs, reward, terminated, truncated, _ = env.step(action)
    rewards.append(reward)
    if terminated or truncated:
        break

total_reward = np.sum(rewards)
print(f"Total Reward: {total_reward:.2f}")

# ===============================
# SAVE VIDEO
# ===============================
if SAVE_VIDEO:
    imageio.mimsave("mlp_policy_rollout.mp4", frames, fps=RENDER_FPS)
    print("Saved rollout video as mlp_policy_rollout.mp4")

# ===============================
# TRAJECTORY PLOT
# ===============================
agents = np.array(agents)
plt.figure(figsize=(6,6))
plt.plot(agents[:,0], agents[:,1], marker='o', markersize=2, label="Agent path")
plt.title("Agent Trajectory under MLP Policy")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.savefig("mlp_policy_trace.png")
plt.show()
