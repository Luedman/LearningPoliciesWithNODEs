import logging
import random

import gym
import torch
from matplotlib import rcParams

# from modules import run_model, HyperParameterWrapper
from nODEDRL.modules import HyperParameterWrapper, run_model
from nODEDRL.visualizations import draw_state

logging.basicConfig(filename='logfile.log', encoding='utf-8', level=logging.DEBUG, filemode='w')

torch.manual_seed(1)
random.seed(1)

rcParams.update({'figure.autolayout': True})
model_experiments = False
if model_experiments:
    for env_name in ['CartPole-v1', 'MountainCarContinuous-v0']:
        env = gym.make(env_name)
        for model_type in ['nODENet']:
            for nodes in [31]:
                state, info = env.reset()
                hp = HyperParameterWrapper(env=env,
                                           model_class_label=model_type,
                                           no_nodes=nodes,
                                           learning_mode='eps_decay_linear',
                                           no_dsteps=10,
                                           epsilon_start=1.0,
                                           epsilon_end=0.05,
                                           learning_rate=0.001,
                                           no_epochs=10,
                                           gamma=0.95,
                                           device_str="cpu",
                                           batch_size=128,
                                           tau=0.005,
                                           label=env_name,
                                           action_dpoints=10)

                run_model(env, hp, run_training=True)
                run_model(env, hp, run_training=False)

draw_state("nODENet_4_128_2_cpu_lr1e-06-y0.999-t0.0005-target-net.pth")
