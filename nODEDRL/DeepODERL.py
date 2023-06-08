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
    for nodes in [16, 32, 64, 128, 256, 512]:
        for lr in [1e-3, 1e-4, 1e-5]:
            state, info = env.reset()

            hp = HyperParameterWrapper(env=env,
                                       no_nodes=nodes,
                                       learning_mode=learning_mode,
                                       no_dsteps=None,
                                       epsilon_start=1.0,
                                       epsilon_end=0.05,
                                       learning_rate=lr,
                                       no_epochs=1500,
                                       device_str="cpu",
                                       label="adjoint")

            # run_model(env, DeepQNet, hp)
            hp.no_dsteps = 10
            run_model(env, nODEnet, hp)
