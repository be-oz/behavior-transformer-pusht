import gymnasium as gym
import gym_pusht

env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
obs, _ = env.reset()
print("Env loaded successfully!")
print("Observation keys:", obs.keys())

frame = env.render()
print("Rendered frame shape:", frame.shape)
