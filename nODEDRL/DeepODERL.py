import random

import gym
import torch

from modules import DQN, runNeuralODE

torch.manual_seed(0)
random.seed(0)

env = gym.make('CartPole-v1')
state, info = env.reset()
n_observations = len(state)
n_actions = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyper_parameter = {
    "gamma": 0.99,
    "tau": 0.005,
    "eps_start": 0.95,
    "eps_end": 0.05,
    "esp_decay": 2500,
    "learning_rate": 1e-5,
    "no_epochs": 20000,
    "batch_size": 128,
    "period_length": 200,
    "device": device,
    "step_size": 1
}

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

# optimizer = torch.optim.AdamW(policy_net.parameters(), lr=hyper_parameter.get("learning_rate"), amsgrad=True)

x_axis_time = torch.linspace(-1, 1, 400, requires_grad=True).to(device)
y_axis = torch.linspace(-1, 1, 400, requires_grad=True).to(device) ** 3
z_axis = torch.sin(y_axis)

runNeuralODE(x_axis_time, y_axis, z_axis, hyper_parameter)
