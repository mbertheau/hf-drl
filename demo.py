from gym.utils.play import PlayPlot
import sys
from stable_baselines3 import PPO
import gym


cum_rew = 0

def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    global cum_rew
    cum_rew += rew
    print(f"Reward: {cum_rew}")
    if terminated or truncated:
        print(f"Resetting cumulative reward, reached {cum_rew} this time.")
        cum_rew = 0
    return [cum_rew]


def demo(model_name):
    algorithm = "PPO"

    env_name = "LunarLander-v2"

    assert model_name.startswith(f"{algorithm}-{env_name}")

    model = PPO.load(model_name)

    plotter = PlayPlot(callback, 30 * 5, ["reward"])

    env = gym.make(env_name, render_mode="human")
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

if __name__ == '__main__':
    demo(sys.argv[1])
    input("Press enter to exit.")
