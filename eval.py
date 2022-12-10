#!/usr/bin/env python3

import common
import sys
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def eval(model_name):
    assert model_name.startswith(f"{common.ALGORITHM}-{common.ENV_NAME}")

    model = PPO.load(model_name)

    env = gym.make(common.ENV_NAME)
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=10, deterministic=True
    )

    score = mean_reward - std_reward
    
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}; score={score}")

if __name__ == '__main__':
    eval(sys.argv[1])
