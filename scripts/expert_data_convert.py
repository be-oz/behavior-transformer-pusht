import numpy as np

INPUT_PATH = "expert_data/expert_trajectories.npy"
OUTPUT_PATH = "transformer_dataset.npy"

def convert():
    raw_trajectories = np.load(INPUT_PATH, allow_pickle=True)
    print(f"📦 Loaded {len(raw_trajectories)} expert trajectories.")

    formatted_data = []
    for traj in raw_trajectories:
        obs_seq = np.array([step["obs"] for step in traj], dtype=np.float32)          # (T, 2)
        action_seq = np.array([step["action"] for step in traj], dtype=np.float32)    # (T, 2)

        formatted_data.append({
            "obs": obs_seq,               # (T, 2)
            "action_bin": np.zeros(len(traj), dtype=np.int64),  # Placeholder, will be filled later
            "residual": action_seq        # (T, 2)
        })

    np.save(OUTPUT_PATH, formatted_data)
    print(f"✅ Saved processed dataset to {OUTPUT_PATH}")

if __name__ == "__main__":
    convert()
