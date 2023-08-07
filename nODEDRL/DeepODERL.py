import random

import gym
import torch
from matplotlib import rcParams

from modules import run_model, HyperParameterWrapper
from models import nODENet, DeepQNet

torch.manual_seed(1)
random.seed(1)

rcParams.update({'figure.autolayout': True})

#env = gym.make('MountainCarContinuous-v0')
env = gym.make('CartPole-v1')

for nodes in [32, 64, 128]:
    for gamma in [0.999, 0.99, 0.95]:
        state, info = env.reset()

        hp = HyperParameterWrapper(env=env,
                                   model_class_label='nODENet',
                                   no_nodes=nodes,
                                   learning_mode='eps_decay_linear',
                                   no_dsteps=10,
                                   epsilon_start=1.0,
                                   epsilon_end=0.0,
                                   learning_rate=0.0001,
                                   no_epochs=100,
                                   gamma=gamma,
                                   device_str="cpu",
                                   batch_size=128,
                                   tau=0.005)

        run_model(env, hp, run_training=True)
        run_model(env, hp, run_training=False)
