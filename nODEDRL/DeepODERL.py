import random

import gym
import torch
from gym.spaces import Box, Discrete

from modules import run_model, ReplayMemory, DeepQNet, nODEnet, HyperParameterWrapper

torch.manual_seed(0)
random.seed(0)

from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

# env = gym.make('MountainCarContinuous-v0')
env = gym.make('CartPole-v1')

state, info = env.reset()

hp = HyperParameterWrapper(env=env,
                           action_dpoints=10,
                           gamma=0.99,
                           tau=0.005,
                           epsilon_start=0.9,
                           epsilon_end=0.05,
                           epsilon_decay=1000,
                           learning_rate=1e-4,
                           no_epochs=2000,
                           batch_size=32,
                           device_str="cuda",
                           period_length=1,
                           label="cuda")

# policy_net = DQN(n_observations, n_actions).to(device)
# target_net = DQN(n_observations, n_actions).to(device)
# target_net.load_state_dict(policy_net.state_dict())

# optimizer = torch.optim.AdamW(policy_net.parameters(), lr=hyper_parameter.get("learning_rate"), amsgrad=True)

#x_axis_time = torch.linspace(-1, 1, 100, requires_grad=True).to(hp.device)
#y_axis = torch.linspace(-1, 1, 100, requires_grad=True).to(device) ** 3
#z_axis = torch.sin(y_axis)

replay_memory = ReplayMemory(10_000)

for model_class in [DeepQNet, nODEnet]:
    run_model(env, model_class, replay_memory, hp)

# runNeuralODE(x_axis_time, y_axis, z_axis, hyper_parameter)
