import random

import gym
import torch
from matplotlib import rcParams

from modules import run_model, HyperParameterWrapper
from models import nODENet, DeepQNet

torch.manual_seed(1)
random.seed(1)

rcParams.update({'figure.autolayout': True})

# env = gym.make('MountainCarContinuous-v0')
env = gym.make('CartPole-v1')

for learning_mode in ['eps_decay_log']:
    for nodes in [8]:
        for lr in [0.1]:
            for tau in [0.005]:
                state, info = env.reset()

                hp = HyperParameterWrapper(env=env,
                                           no_nodes=nodes,
                                           learning_mode=learning_mode,
                                           no_dsteps=None,
                                           epsilon_start=0.9,
                                           epsilon_end=0.05,
                                           learning_rate=lr,
                                           no_epochs=2000,
                                           device_str="cpu",
                                           batch_size=128)

                run_model(env, DeepQNet, hp)
                hp.no_dsteps = 10
                # run_model(env, nODEnet, hp)
