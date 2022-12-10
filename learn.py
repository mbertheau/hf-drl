from simple_term_menu import TerminalMenu
import sys
import gym
import gym.envs.box2d.lunar_lander
from gym.utils.play import PlayPlot

# gym.envs.box2d.lunar_lander.INITIAL_RANDOM = 10
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


"""
Possible strategy:
train several models from scratch, pick one that converges quickly
i.e. always train 10 in parallel
select quickest converging and train 10 copies

"""


cum_rew = 0


def demo_model(model_name):
    global cum_rew
    cum_rew = 0

    def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
        global cum_rew
        cum_rew += rew
        print(f"Reward: {cum_rew}")
        if terminated or truncated:
            print(f"Resetting cumulative reward, reached {cum_rew} this time.")
            cum_rew = 0
        return [cum_rew]

    plotter = PlayPlot(callback, 30 * 5, ["reward"])

    env = gym.make(env_name, render_mode="human")
    model = models[model_name]
    print(f"Demoing {model_name}")
    obs, info = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

        plotter.callback(obs, obs, action, reward, terminated, truncated, info)

        if terminated:
            print("Terminated.")
            break

        if truncated:
            print("Truncated.")
            break

    env.close()
    del env
    del plotter


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
