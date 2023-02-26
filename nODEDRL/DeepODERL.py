import random

import gym
import torch
from gym.spaces import Box, Discrete

from modules import runNeuralODE_gym, ReplayMemory

torch.manual_seed(0)
random.seed(0)

from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

env = gym.make('MountainCarContinuous-v0')
#env = gym.make('CartPole-v1')

state, info = env.reset()

observations_low, observations_high = env.observation_space.low, env.observation_space.high
if type(env.action_space) is Box:
    action_low, action_high = env.action_space.low, env.action_space.high
elif type(env.action_space) is Discrete:
    action_low, action_high = (0, env.action_space.n)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

hyper_parameter = {
    "gamma": 0.99,
    "tau": 0.005,
    "eps_start": 0.95,
    "eps_end": 0.05,
    "eps_decay": 250,
    "learning_rate": 1e-4,
    "no_epochs": 500,
    "batch_size": 128,
    "period_length": 50,
    "device": device,
    "step_size": 1,
    "no_sdpoints": 100,
    "no_adpoints": 10,
    "observations_low": observations_low,
    "observations_high": observations_high,
    "action_low": action_low,
    "action_high": action_high
}

# policy_net = DQN(n_observations, n_actions).to(device)
# target_net = DQN(n_observations, n_actions).to(device)
# target_net.load_state_dict(policy_net.state_dict())

# optimizer = torch.optim.AdamW(policy_net.parameters(), lr=hyper_parameter.get("learning_rate"), amsgrad=True)

x_axis_time = torch.linspace(-1, 1, 100, requires_grad=True).to(device)
y_axis = torch.linspace(-1, 1, 100, requires_grad=True).to(device) ** 3
z_axis = torch.sin(y_axis)

replay_memory = ReplayMemory(500)
runNeuralODE_gym(env, replay_memory, hyper_parameter)

# runNeuralODE(x_axis_time, y_axis, z_axis, hyper_parameter)
