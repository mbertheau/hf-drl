from simple_term_menu import TerminalMenu
import sys
import gym
import gym.envs.box2d.lunar_lander
from gym.utils.play import PlayPlot

# gym.envs.box2d.lunar_lander.INITIAL_RANDOM = 10
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy

env_name = "LunarLander-v2"

n_steps = 100_000

TORCH_DEVICE = "cpu"


models = {
    "A2C": A2C("MlpPolicy", env_name, device=TORCH_DEVICE),
    "PPO": PPO("MlpPolicy", env_name, device=TORCH_DEVICE, verbose=1),
    "PPO2": PPO(
        "MlpPolicy",
        env_name,
        device=TORCH_DEVICE,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1,
    ),
}


def find_good_model(model_name):
    model_class = {"A2C": A2C, "PPO": PPO, "PPO2": PPO}
    candidate_models = [
        model_class[model_name]("MlpPolicy", env_name, device=TORCH_DEVICE, verbose=1)
        for i in range(10)
    ]

    best_score = -1000000
    best_model = None

    for i, model in enumerate(candidate_models):
        model.learn(2000)
        score = eval_model("", model)

        if score > best_score:
            print(f"New best model score: {score}")
            best_score = score
            best_model = model

    models[model_name] = best_model


def train_model(model_name):
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


def save_model(model_name):
    fname = input("Name: ")
    models[model_name].save(fname)


def eval_model(model_name, model=None):
    if not model:
        model = models[model_name]
    env = gym.make(env_name)
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=10, deterministic=True
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    return mean_reward - std_reward


class Return(Exception):
    pass


def do_return():
    raise Return


def model_menu(model_name):
    options = ["Find good", "Train", "Evaluate", "Demo", "Save", "Exit"]
    actions = [
        lambda: find_good_model(model_name),
        lambda: train_model(model_name),
        lambda: eval_model(model_name),
        lambda: demo_model(model_name),
        lambda: save_model(model_name),
        do_return,
    ]

    while True:
        try:
            actions[TerminalMenu(options).show()]()
        except Return:
            return


def main():
    actions = []
    options = []

    for model_name in models.keys():
            options.append(f"Select model {model_name}")
            actions.append(lambda model_name=model_name: model_menu(model_name))

    options.append("Exit")
    actions.append(lambda: sys.exit(0))

    while True:
        actions[TerminalMenu(options).show()]()


main()
