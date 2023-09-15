import logging
import random
import sys
from multiprocessing import Pool

import gym
import torch
from matplotlib import rcParams

from modules import HyperParameterWrapper, eval_model, run_training

logging.basicConfig(filename='logfile.log', encoding='utf-8', level=logging.DEBUG, filemode='w')

torch.manual_seed(1)
random.seed(1)

in_colab = 'google.colab' in sys.modules
rcParams.update({'figure.autolayout': True})

env_label = "CartPole-v1"  # 'MountainCarContinuous-v0';.
hp1 = HyperParameterWrapper(env_label=env_label,
                            model_class_label='DeepQNet',
                            no_nodes=128,
                            learning_mode='eps_decay_linear',
                            no_dsteps=60,
                            epsilon_start=0.5,
                            epsilon_end=0.00,
                            learning_rate=0.0005,
                            no_epochs=2000,
                            gamma=0.99,
                            device_str="cpu",
                            batch_size=512,
                            tau=0.005,
                            label=env_label + "-d60",
                            action_dpoints=5,
                            colab=in_colab)

hp2 = HyperParameterWrapper(env_label=env_label,
                            model_class_label='DeepQNet',
                            no_nodes=128,
                            learning_mode='eps_decay_linear',
                            no_dsteps=30,
                            epsilon_start=0.5,
                            epsilon_end=0.00,
                            learning_rate=0.0005,
                            no_epochs=2000,
                            gamma=0.99,
                            device_str="cpu",
                            batch_size=512,
                            tau=0.005,
                            label=env_label + "-d30-lb",
                            action_dpoints=5,
                            colab=in_colab)

hp3 = HyperParameterWrapper(env_label=env_label,
                            model_class_label='DeepQNet',
                            no_nodes=128,
                            learning_mode='eps_decay_linear',
                            no_dsteps=30,
                            epsilon_start=0.0,
                            epsilon_end=0.00,
                            learning_rate=0.0005,
                            no_epochs=2000,
                            gamma=0.99,
                            device_str="cpu",
                            batch_size=512,
                            tau=0.005,
                            label=env_label + "-d30-oP",
                            action_dpoints=5,
                            colab=in_colab)


def run_experiment(hp):
    hp.env = gym.make(env_label)
    run_training(hp)
    hp.no_epochs = 1500
    eval_model(hp, fixed_epsilon=0.0)
    eval_model(hp, fixed_epsilon=0.05)
    eval_model(hp, fixed_epsilon=0.1)
    eval_model(hp, fixed_epsilon=0.2)
    eval_model(hp, fixed_epsilon=0.5)


if __name__ == '__main__':
    pool = Pool(processes=2)
    for hp in [hp1, hp2, hp3]:
        pool.apply_async(run_experiment, args=(hp,))
    pool.close()
    pool.join()

# draw_state("nODENet_4_128_2_cpu_lr1e-06-y0.999-t0.0005-target-net.pth")
