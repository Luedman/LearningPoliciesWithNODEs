import random

import gym
import torch
from matplotlib import rcParams

from modules import run_model, ReplayMemory, nODEnet, HyperParameterWrapper, DeepQNet

torch.manual_seed(1)
random.seed(1)

rcParams.update({'figure.autolayout': True})

# env = gym.make('MountainCarContinuous-v0')
env = gym.make('CartPole-v1')

for learning_mode in ['eps_decay_linear']:
    for nodes in [16, 32, 64, 128, 256]:
        state, info = env.reset()

        hp = HyperParameterWrapper(env=env,
                                   no_nodes=nodes,
                                   learning_mode=learning_mode,
                                   no_dsteps=10,
                                   epsilon_start=0.95,
                                   epsilon_end=0.05,
                                   learning_rate=1e-4,
                                   no_epochs=1000,
                                   device_str="cpu")

        run_model(env, DeepQNet, hp)

        for dpoints in [10, 100]:
            replay_memory = ReplayMemory(1_000)
            hp = HyperParameterWrapper(env=env,
                                       no_nodes=nodes,
                                       learning_mode=learning_mode,
                                       no_dsteps=dpoints,
                                       epsilon_start=0.95,
                                       epsilon_end=0.05,
                                       learning_rate=1e-4,
                                       no_epochs=1000,
                                       device_str="cpu")
            run_model(env, nODEnet, hp)
