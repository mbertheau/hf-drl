from simple_term_menu import TerminalMenu
import sys
import gym
from stable_baselines3 import A2C, PPO

env_name = "LunarLander-v2"

n_steps = 100_000

models = {
    "A2C": A2C("MlpPolicy", env_name, verbose=1),
    "PPO": PPO("MlpPolicy", env_name, verbose=1),
    "PPO2": PPO(
        "MlpPolicy",
        env_name,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1,
    ),
}


def train_more(model_name):
    print(f"Training {model_name} model for {n_steps} steps")
    models[model_name] = models[model_name].learn(n_steps)


def demo(model_name):
    env = gym.make(env_name, render_mode="human")
    model = models[model_name]
    print(f"Demoing {model_name}")
    obs, info = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

        if terminated:
            print("Terminated.")
            break

        if truncated:
            print("Truncated.")
            break

    env.close()


actions = []
options = []

for model_name in models.keys():
    options.append(f"Train {model_name} for {n_steps} more steps")
    actions.append(lambda model_name=model_name: train_more(model_name))
    options.append(f"Demo {model_name}")
    actions.append(lambda model_name=model_name: demo(model_name))

options.append("Exit")
actions.append(lambda: sys.exit(0))

while True:
    actions[TerminalMenu(options).show()]()
