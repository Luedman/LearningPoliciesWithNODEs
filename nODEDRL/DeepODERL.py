import logging
import random

import gym
import torch
from matplotlib import rcParams

from modules import run_model, HyperParameterWrapper

logging.basicConfig(filename='logfile.log', encoding='utf-8', level=logging.DEBUG, filemode='w')

torch.manual_seed(1)
random.seed(1)

rcParams.update({'figure.autolayout': True})

env = gym.make('MountainCarContinuous-v0')
# env = gym.make('CartPole-v1')


for model_type in ['nODENet']:
    for nodes in [32]:
        state, info = env.reset()

        hp = HyperParameterWrapper(env=env,
                                   model_class_label=model_type,
                                   no_nodes=nodes,
                                   learning_mode='eps_decay_linear',
                                   no_dsteps=10,
                                   epsilon_start=1.0,
                                   epsilon_end=0.05,
                                   learning_rate=0.001,
                                   no_epochs=1000,
                                   gamma=0.95,
                                   device_str="cpu",
                                   batch_size=128,
                                   tau=0.005,
                                   label='mountain-car',
                                   action_dpoints=10)

        run_model(env, hp, run_training=True)
        run_model(env, hp, run_training=False)
