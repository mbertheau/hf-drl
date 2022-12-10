ENV_NAME = "LunarLander-v2"
ALGORITHM = "PPO"
HYPERPARAMS = {"n_steps": 1024,
               "batch_size": 64,
               "gae_lambda": 0.98,
               "gamma": 0.999,
               "n_epochs": 4,
               "ent_coef": 0.01}
