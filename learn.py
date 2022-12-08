import gym
from stable_baselines3 import A2C

env = gym.make("LunarLander-v2")
model = A2C("MlpPolicy", env).learn(10000)

env = gym.make("LunarLander-v2", render_mode="human")
obs, info = env.reset()
print(f"Observation after reset: {obs}")
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
