import matplotlib.pyplot as plt
import numpy as np
import torch
from my_policy import MyPolicy
import gymnasium as gym
import gym_pusht

# --- Load expert data ---
data = np.load("expert_data/expert_dataset.npy", allow_pickle=True)
expert_obs = data[0]["obs"]  # (T, 2)

# --- Run policy ---
policy = MyPolicy()
env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", max_episode_steps=len(expert_obs))
obs, _ = env.reset(seed=42)

policy_traj = [obs["agent_pos"][:2]]
done = False
while not done:
    state = torch.tensor(obs["agent_pos"][:2]).float().unsqueeze(0)
    image = torch.tensor(obs["pixels"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    action = policy.act({"observation.state": state, "observation.image": image})
    obs, _, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
    done = terminated or truncated
    policy_traj.append(obs["agent_pos"][:2])

policy_traj = np.array(policy_traj)

# --- Plot ---
plt.figure(figsize=(6, 6))
plt.plot(expert_obs[:, 0], expert_obs[:, 1], 'b.-', label="Expert")
plt.plot(policy_traj[:, 0], policy_traj[:, 1], 'r.-', label="Policy")
plt.title("Expert vs Policy Trajectory")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.legend()

plt.savefig("outputs/expert_vs_policy_ep0.png")
print("📈 Saved: outputs/expert_vs_policy_ep0.png")
