import gym
from stable_baselines3 import A2C, PPO

env_name = "LunarLander-v2"

n_steps = 50_000

models = {}

print("Training A2C model")
models["A2C"] = A2C("MlpPolicy", env_name).learn(n_steps)

print("Training PPO model")
models["PPO"] = PPO("MlpPolicy", env_name).learn(n_steps)

env = gym.make(env_name, render_mode="human")
for model_name, model in models.items():
    print(f"Demoing {model_name}")
    obs, info = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

        if terminated or truncated:
            break
