import logging
import random
import sys
from multiprocessing import Process

import gym
import torch
from matplotlib import rcParams

from modules import run_model, HyperParameterWrapper

logging.basicConfig(filename='logfile.log', encoding='utf-8', level=logging.DEBUG, filemode='w')

torch.manual_seed(1)
random.seed(1)

in_colab = 'google.colab' in sys.modules
rcParams.update({'figure.autolayout': True})

env_label = 'MountainCarContinuous-v0'
hp1 = HyperParameterWrapper(env_label=env_label,
                            model_class_label='nODENet',
                            no_nodes=128,
                            learning_mode='eps_decay_linear',
                            no_dsteps=30,
                            epsilon_start=1.0,
                            epsilon_end=0.00,
                            learning_rate=0.0001,
                            no_epochs=3000,
                            gamma=0.99,
                            device_str="cpu",
                            batch_size=128,
                            tau=0.005,
                            label=env_label + "-d30",
                            action_dpoints=5,
                            colab=in_colab)

hp2 = HyperParameterWrapper(env_label=env_label,
                            model_class_label='nODENet',
                            no_nodes=128,
                            learning_mode='eps_decay_linear',
                            no_dsteps=5,
                            epsilon_start=1.0,
                            epsilon_end=0.00,
                            learning_rate=0.0001,
                            no_epochs=3000,
                            gamma=0.99,
                            device_str="cpu",
                            batch_size=128,
                            tau=0.005,
                            label=env_label + "-d5",
                            action_dpoints=5,
                            colab=in_colab)


def run_experiment(hp):
    hp.env = gym.make(env_label)
    run_model(hp, run_training=True, start_episode=1)
    run_model(hp, run_training=False)


if __name__ == '__main__':
    experiments_list = []
    for hp in [hp1, hp2]:
        experiment = Process(target=run_experiment, args=(hp,))
        experiments_list.append(experiment)
        experiment.start()

    for experiment in experiments_list:
        experiment.join()
