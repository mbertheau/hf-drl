import gym
import time

env = gym.make("LunarLander-v2")

import gym.envs.box2d.lunar_lander
gym.envs.box2d.lunar_lander.INITIAL_RANDOM=10

observation = env.reset()
env.render()

for _ in range(20):
    action = env.action_space.sample()

    observation, reward, done, info = env.step(action)
    env.render()

    time.sleep(1)

    print(f"Action taken: {action}")
    print(f"Reward: {reward}")
    print(f"Info: {info}")
    print(f"New observation: {observation}")

    if done:
        break


print("Done.")
print(f"Last observation: {observation}")