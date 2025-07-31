import matplotlib.pyplot as plt
import numpy as np

data = np.load("expert_data/expert_dataset.npy", allow_pickle=True)

plt.figure(figsize=(6, 6))
plt.scatter(data[0]["obs"][:, 0], data[0]["obs"][:, 1], c='blue', s=10)
plt.title("Expert Trajectory (Episode 0)")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)

plt.savefig("outputs/expert_trajectory_ep0.png")
print("Saved trajectory plot to outputs/expert_trajectory_ep0.png")
