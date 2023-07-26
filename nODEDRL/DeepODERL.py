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
    for nodes in [64, 128, 256]:
        for lr in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.00001, 0.000005, 0.000001]:
            for tau in [0.01, 0.005, 0.001, 0.0005, 0.00001, 0.000005, 0.000001]:
                for gamma in [0.99, 0.95, 0.9]:
                    for batch_size in [32, 64, 128, 254]:
                        state, info = env.reset()

                        hp = HyperParameterWrapper(env=env,
                                                   model_class_label='DeepQNet',
                                                   no_nodes=nodes,
                                                   learning_mode=learning_mode,
                                                   no_dsteps=None,
                                                   epsilon_start=0.9,
                                                   epsilon_end=0.05,
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
