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

for learning_mode in ['eps_decay_linear']:
    for nodes in [32, 64, 128]:
        for lr in [1e-3, 1e-4, 1e-5, 1e-6]:
            for tau in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
                for gamma in [0.999, 0.99, 0.95]:
                    for batch_size in [128]:
                        state, info = env.reset()

                        hp = HyperParameterWrapper(env=env,
                                                   model_class_label='nODENet',
                                                   no_nodes=nodes,
                                                   learning_mode=learning_mode,
                                                   no_dsteps=10,
                                                   epsilon_start=1.0,
                                                   epsilon_end=0.0,
                                                   learning_rate=lr,
                                                   no_epochs=2000,
                                                   gamma=gamma,
                                                   device_str="cpu",
                                                   batch_size=128,
                                                   tau=tau)

                        run_model(env, hp, run_training=True)
                        run_model(env, hp, run_training=False)
                        break
                    break
                # hp.no_dsteps = 10
                # run_model(env, nODEnet, hp)
