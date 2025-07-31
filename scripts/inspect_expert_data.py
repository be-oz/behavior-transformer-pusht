import numpy as np

data = np.load("expert_data/expert_trajectories.npy", allow_pickle=True)
print(f"Total trajectories: {len(data)}")

# Print the type and structure of the first element
print("\nFirst trajectory type:", type(data[0]))
print("First trajectory content:")
print(data[0])
