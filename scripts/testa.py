import gymnasium as gym
import gym_pusht  # ✅ This is needed to register the env

env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
obs, _ = env.reset()
print("Env loaded successfully!")
print("Observation keys:", obs.keys())  # should print: dict_keys(['pixels', 'agent_pos'])

frame = env.render()
print("Rendered frame shape:", frame.shape)
