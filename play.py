import time

import gym

env = gym.make("LunarLander-v2", render_mode="rgb_array")
from gym.utils.play import play, PlayPlot
import gym.envs.box2d.lunar_lander
import pygame


gym.envs.box2d.lunar_lander.INITIAL_RANDOM = 10

cum_rew = 0


def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    global cum_rew
    cum_rew += rew
    if terminated or truncated:
        print(f"Resetting cumulative reward, reached {cum_rew} this time.")
        cum_rew = 0
    return [cum_rew]


plotter = PlayPlot(callback, 30 * 5, ["reward"])

play(
    env,
    keys_to_action={
        (pygame.K_1,): 1,
        (pygame.K_3,): 3,
        (pygame.K_2,): 2,
        (pygame.K_4,): 0,
    },
    callback=plotter.callback,
)
