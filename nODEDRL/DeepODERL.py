import random

import gym
import torch

from modules import DQN, runNeuralODE, runNeuralODE_gym, ReplayMemory

torch.manual_seed(0)
random.seed(0)

env = gym.make('MountainCarContinuous-v0')
state, info = env.reset()

observations_low, observations_high = env.observation_space.low, env.observation_space.high
action_low, action_high = env.action_space.low, env.action_space.high

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


hyper_parameter = {
    "gamma": 0.99,
    "tau": 0.005,
    "eps_start": 0.99,
    "eps_end": 0.05,
    "eps_decay": 2_500,
    "learning_rate": 1e-4,
    "no_epochs": 20_000,
    "batch_size": 8,
    "period_length": 50,
    "device": device,
    "step_size": 1
}

#policy_net = DQN(n_observations, n_actions).to(device)
#target_net = DQN(n_observations, n_actions).to(device)
#target_net.load_state_dict(policy_net.state_dict())

# optimizer = torch.optim.AdamW(policy_net.parameters(), lr=hyper_parameter.get("learning_rate"), amsgrad=True)

x_axis_time = torch.linspace(-1, 1, 100, requires_grad=True).to(device)
y_axis = torch.linspace(-1, 1, 100, requires_grad=True).to(device) ** 3
z_axis = torch.sin(y_axis)

replay_memory = ReplayMemory(500)
runNeuralODE_gym(env, replay_memory, hyper_parameter)

#runNeuralODE(x_axis_time, y_axis, z_axis, hyper_parameter)
