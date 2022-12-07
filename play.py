import time

import gym

env = gym.make("LunarLander-v2")
from gym.utils.play import play, PlayPlot
import gym.envs.box2d.lunar_lander
import pygame


gym.envs.box2d.lunar_lander.INITIAL_RANDOM = 10


def callback(obs_t, obs_tp1, action, rew, done, info):
    return [rew]


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
