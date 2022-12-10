#!/usr/bin/env python3

"""
Train 10 new models or descendants of an existing model
"""

import common
from datetime import datetime
import gym
import sys
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

N_STEPS = 2_000_000


hyperparams = {"n_steps": 1024,
               "batch_size": 64,
               "gae_lambda": 0.98,
               "gamma": 0.999,
               "n_epochs": 4,
               "ent_coef": 0.01}

def train(model_name):
    for i in range(10):
        if model_name:
            assert model_name.startswith(f"{common.ALGORITHM}-{common.ENV_NAME}")

            save_name = model_name

            if save_name.endswith('.zip'):
                save_name = save_name[:-4]

            model = PPO.load(model_name, env=gym.make(common.ENV_NAME))
        else:
            model = PPO("MlpPolicy", common.ENV_NAME, **hyperparams)
            save_name = f"{common.ALGORITHM}-{common.ENV_NAME}-{datetime.now().isoformat()}"

        model = model.learn(N_STEPS)

        env = gym.make(common.ENV_NAME)
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=10, deterministic=True
        )

        print(datetime.now())

        score = mean_reward - std_reward

        print(f"Model {i}: mean_reward={mean_reward:.2f} +/- {std_reward}; score={score:.1f}")


        model.save(f"{save_name}-{i}-{score:.1f}")


if __name__ == '__main__':

    model_name = None
    if len(sys.argv) > 1:
        model_name = sys.argv[1]

    train(model_name)
