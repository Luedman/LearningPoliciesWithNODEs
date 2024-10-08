import logging
import math
import os
import random
import time
from collections import namedtuple, deque
from datetime import datetime
from itertools import count
from operator import itemgetter

import gym
import matplotlib
import numpy as np
import torch
from gym.spaces import Box, Discrete
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from models import nODENet, DeepQNet

is_ipython = 'inline' in matplotlib.get_backend()

Transition = namedtuple("Transition", ('state', 'action', 'next_state', 'reward'))


class HyperParameterWrapper:
    def __init__(self,
                 env_label,
                 epsilon_start: float,
                 model_class_label: str,
                 epsilon_end: float,
                 learning_rate: float,
                 no_epochs: int,
                 no_dsteps: int = 10,
                 no_nodes: int = 32,
                 epsilon_decay: int = None,
                 learning_mode="off-policy",
                 batch_size: int = 256,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 period_length: int = 1,
                 device_str: str = 'cpu',
                 action_dpoints: int = 10,
                 label: str = None,
                 colab=False,
                 fixed_epsilon: float = 0.0):
        self.env = gym.make(env_label)
        self.model_class_label = model_class_label
        self.obs_high = self.env.observation_space.high
        self.obs_low = self.env.observation_space.low
        self.no_observations = len(self.obs_high)
        self.action_dpoints = action_dpoints
        self.gamma = gamma
        self.tau = tau
        self.learning_mode = learning_mode
        self.no_nodes = no_nodes
        self.no_dsteps = no_dsteps

        self.eps_start = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay or no_epochs * 2
        self.fixed_epsilon = fixed_epsilon

        self.learning_rate = learning_rate
        self.no_epochs = no_epochs
        self.batch_size = batch_size

        self.period_length = period_length
        self.label = f'-{label}' if label is not None else ''
        self.colab = colab

        if device_str is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        if type(self.env.action_space) is Box:
            self.action_type = "box"
            self.action_low, self.action_high = self.env.action_space.low, self.env.action_space.high
            self.disc_action_space = torch.linspace(self.action_low[0],
                                                    self.action_high[0],
                                                    self.action_dpoints, device=self.device)

        elif type(self.env.action_space) is Discrete:
            self.action_type = "discrete"
            self.action_low, self.action_high = (0, self.env.action_space.n)
            self.disc_action_space = list(range(self.env.action_space.n))
            self.action_dpoints = self.env.action_space.n

    @property
    def model_label(self) -> str:
        return f"{self.model_class_label}_{self.no_observations}_{self.no_nodes}_{self.action_dpoints}_{self.device}"

    @property
    def model_training_label(self) -> str:
        return self.model_label + f"_lr{str(self.learning_rate)}-y{self.gamma}-t{self.tau}" + self.label

    def epsilon_threshold(self, epoch) -> float:
        if epoch is None:
            eps_threshold = self.fixed_epsilon
        elif self.learning_mode == 'eps_decay_log':
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * epoch / self.eps_decay)
        elif self.learning_mode == 'eps_decay_linear':
            eps_threshold = max([self.eps_start - (self.eps_start - self.eps_end) * (epoch / self.no_epochs) * 2,
                                 self.eps_end])
        elif self.learning_mode == 'off-policy':
            if epoch / self.no_epochs < 0.9:
                eps_threshold = self.eps_start
            else:
                eps_threshold = 0.0
        elif self.learning_mode == 'on-policy':
            if epoch / self.no_epochs < 0.9:
                eps_threshold = self.eps_end
            else:
                eps_threshold = 0.0
        else:
            eps_threshold = 0.0
        return eps_threshold

    def conv_action(self, action):
        if self.action_type == "box":
            return action.cpu().numpy()
        elif self.action_type == "discrete":
            return int(action)

    def get_action_index(self, action_value) -> torch.tensor:
        if self.action_type == "box":
            action_index = torch.bucketize(action_value, self.disc_action_space)
        elif self.action_type == "discrete":
            action_index = torch.bucketize(action_value, torch.tensor(self.disc_action_space, device=self.device))
        return action_index


class ReplayMemory(Dataset):
    def __init__(self, capacity: int, alpha=0.5):
        self._memory = deque([], maxlen=capacity)
        self._priorities = deque([], maxlen=capacity)
        self._probabilities = deque([], maxlen=capacity)
        self._alpha = alpha

    def __len__(self):
        return len(self._memory)

    def __getitem__(self, idx):
        return self._memory[idx]

    def push(self, state, action, next_state, reward, priority=1):
        self._memory.append(Transition(state, action, next_state, reward))
        self._priorities.append(priority)

    def sample(self, batch_size):
        idx_choice = random.choices(range(len(self._memory)), weights=self.probabilities, k=batch_size)
        elements = list(itemgetter(*idx_choice)(self._memory))
        return idx_choice, elements

    def get_memory(self):
        return self._memory

    def update_priorities(self, indices, priorities):
        for choice_idx, memory_idx in enumerate(indices):
            self._priorities[memory_idx] = float(priorities[choice_idx])

    @property
    def probabilities(self):
        self._probabilities = np.power(np.array(self._priorities), self._alpha)
        return np.divide(self._probabilities, np.sum(self._probabilities))


def select_action(state: torch.tensor, epoch, policy_net: torch.nn.Module,
                  hp: HyperParameterWrapper) -> (torch.tensor, float):
    eps_threshold = hp.epsilon_threshold(epoch)
    with torch.no_grad():
        rand = random.random()
        if rand > eps_threshold:
            action_values = policy_net(state.T)
            action_idx = action_values.argmax()
        else:
            action_idx = torch.multinomial(torch.ones(len(hp.disc_action_space)), 1)
        action_idx_tensor = torch.tensor(action_idx,
                                         device=hp.device,
                                         dtype=torch.int64).reshape(1, )
    return action_idx_tensor, eps_threshold


def make_tensorboard_writer(folder: str, run_label: str) -> SummaryWriter:
    writer = SummaryWriter(log_dir=f"runs/{folder}/{run_label}")
    return writer


def training_step(replay_memory, hp, policy_net, target_net, optimizer, scheduler, writer, total_steps):
    loss, target_net, policy_net = optimize_model(replay_memory, hp, policy_net, target_net,
                                                  optimizer, scheduler)

    writer.add_scalar('Loss/train', loss.item(), total_steps)
    return loss, target_net, policy_net, writer


def write_tensorboard(writer, epoch, total_reward_per_epoch, eps_threshold,
                      no_solves, steps, action_values, total_steps, start_time):
    writer.add_scalar('Reward/train', total_reward_per_epoch, epoch)
    writer.add_scalar('Epsilon Threshold/train', eps_threshold, epoch)
    writer.add_scalar('No. Solves', no_solves, epoch)
    writer.add_scalar('Steps', steps, epoch)
    writer.add_scalar('Avg. Action value', action_values / total_steps, epoch)
    writer.add_scalar('Time/Epoch', time.time() - start_time, epoch)
    return writer


def load_model(hp, label="policy-net") -> torch.nn.Module:
    model_file_name = hp.model_training_label + f"-{label}.pth"
    (model_type, n_observations, no_nodes, n_actions, device, _) = model_file_name.split('_')
    path = os.path.join(os.getcwd(), "models", model_file_name)
    if model_type == "nODENet":
        model = nODENet(int(n_observations), int(n_actions), int(no_nodes), device, 10)
        model.load_state_dict(torch.load(path))
    elif model_type == "DeepQNet":
        model = DeepQNet(int(n_observations), int(n_actions), int(no_nodes), device)
        model.load_state_dict(torch.load(path))
    else:
        raise ValueError("model_type unknown")
    return model


def init_model(hp: HyperParameterWrapper) -> (ReplayMemory, torch.nn.Module, torch.nn.Module, AdamW, ReduceLROnPlateau):
    replay_memory = ReplayMemory(10_000)

    if hp.model_class_label == 'nODENet':
        model_class = nODENet
    elif hp.model_class_label == 'DeepQNet':
        model_class = DeepQNet
    else:
        raise ValueError("model_class_label not known")

    policy_net = model_class(n_observations=hp.no_observations,
                             n_actions=hp.action_dpoints,
                             device=hp.device,
                             no_nodes=hp.no_nodes,
                             no_dsteps=hp.no_dsteps)
    target_net = model_class(n_observations=hp.no_observations,
                             n_actions=hp.action_dpoints,
                             device=hp.device,
                             no_nodes=hp.no_nodes,
                             no_dsteps=hp.no_dsteps)

    optimizer = AdamW(policy_net.parameters(), lr=hp.learning_rate, amsgrad=True)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=10_000, min_lr=1e-6,
                                  cooldown=10_000, factor=0.5)

    return replay_memory, policy_net, target_net, optimizer, scheduler


def run_training(hp: HyperParameterWrapper, start_episode: int = 1):
    replay_memory, policy_net, target_net, optimizer, scheduler = init_model(hp)
    writer = make_tensorboard_writer('train', hp.model_training_label)

    if start_episode > 1:
        policy_net = load_model(hp)
        target_net = load_model(hp, "target-net")
    if not os.path.exists(os.path.join(os.getcwd(), "models")):
        os.mkdir(os.path.join(os.getcwd(), "models"))

    run_simulation(True, replay_memory, policy_net, target_net, optimizer, scheduler, start_episode, hp, writer)


def eval_model(hp: HyperParameterWrapper, fixed_epsilon: float = 0.0):
    hp.fixed_epsilon = fixed_epsilon
    replay_memory, policy_net, target_net, optimizer, scheduler = None, load_model(hp), None, None, None
    writer = make_tensorboard_writer('eval', hp.model_training_label + f'-eps{fixed_epsilon}')
    run_simulation(False, replay_memory, policy_net, target_net, optimizer, scheduler, 1, hp, writer)


def run_simulation(run_training, replay_memory, policy_net, target_net, optimizer, scheduler, start_episode, hp,
                   writer):
    loss_per_epoch, reward_per_epoch, avg_loss, avg_reward = [], [], [], []
    no_solves, total_steps, action_values_cumulated, loss = 0, 0 , 0, 0
    start_time_total = time.time()
    for epoch in range(start_episode, hp.no_epochs + 1):
        start_time = time.time()
        state, info = hp.env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32, device=hp.device)[:, None, None]
        total_reward_per_epoch = 0
        for steps in count():
            total_steps += 1
            if ('MountainCarContinuous-v0' in hp.label and (steps == 0 or steps % 5 == 0)) or \
                    ('CartPole-v1' in hp.label):
                epoch_ = epoch if run_training else None
                action_idx_tensor, eps_threshold = select_action(state_tensor, epoch_, policy_net=policy_net, hp=hp)

            action_value = hp.disc_action_space[action_idx_tensor]
            observation, reward, terminated, truncated, _ = hp.env.step(hp.conv_action(action_value))

            done = (terminated or truncated)

            if truncated:
                no_solves += 1 if 'CartPole-v1' in hp.label else 0
                next_state_tensor = None
            elif terminated:
                no_solves += 1 if 'MountainCarContinuous-v0' in hp.label else 0
                next_state_tensor = None
            else:
                observation_tensor = torch.tensor(observation, dtype=torch.float32, device=hp.device,
                                                  requires_grad=True)
                next_state_tensor = torch.reshape(observation_tensor, (len(hp.obs_high), 1, 1))
                steps += 1

            total_reward_per_epoch += reward
            reward_tensor = torch.tensor([reward], device=hp.device)
            if run_training:
                replay_memory.push(state_tensor, action_idx_tensor, next_state_tensor, reward_tensor)
                if len(replay_memory) > hp.batch_size:
                    loss, target_net, policy_net = optimize_model(replay_memory,
                                                                  hp,
                                                                  policy_net,
                                                                  target_net,
                                                                  optimizer,
                                                                  scheduler)

                    loss_per_epoch.append(loss.item())
                    avg_loss.append(torch.mean(torch.mean(torch.tensor(loss_per_epoch[-20:]))))

                    writer.add_scalar('Loss/train', loss.item(), total_steps)
                    writer.add_scalar('Loss/train_avg', torch.mean(torch.tensor(loss_per_epoch[-200:])),
                                      total_steps)

            state_tensor = next_state_tensor

            if done:
                reward_per_epoch.append(total_reward_per_epoch)
                writer.add_scalar('Reward/train', total_reward_per_epoch, epoch)
                writer.add_scalar('Reward/train_avg', torch.mean(torch.tensor(reward_per_epoch[-20:])), epoch)
                writer.add_scalar('Epsilon Threshold/train', eps_threshold, epoch)
                writer.add_scalar('No. Solves', no_solves, epoch)
                writer.add_scalar('Steps', steps, epoch)
                writer.add_scalar('Time/Epoch', time.time() - start_time, epoch)
                writer.add_scalar('Time/Total', time.time() - start_time_total, epoch)
                mode = 'train' if run_training else 'eval'
                episode_summary = f"{datetime.now().strftime('%H:%M:%S')} {mode} E{epoch}: Done after {steps}" + \
                                  f" steps, terminated: {terminated}, truncated: {truncated}," + \
                                  f" reward: {total_reward_per_epoch:.2f}, time: {time.time() - start_time:.2f}" + \
                                  f" sec, eps: {eps_threshold:.2f}, loss {loss:.2f}, {hp.model_training_label}"
                logging.info(episode_summary)
                if epoch % 25 == 0:
                    print(episode_summary)
                    if run_training:
                        model_label_tn = f"{hp.model_training_label}-target-net.pth"
                        model_label_pn = f"{hp.model_training_label}-policy-net.pth"
                        torch.save(target_net.state_dict(), os.path.join(os.getcwd(), "models", model_label_tn))
                        torch.save(policy_net.state_dict(), os.path.join(os.getcwd(), "models", model_label_pn))
                break


def optimize_model(replay_memory: ReplayMemory,
                   hp: HyperParameterWrapper,
                   policy_net: torch.nn.Module,
                   target_net: torch.nn.Module,
                   optimizer: torch.optim,
                   scheduler: torch.optim):
    idx_choice, transitions = replay_memory.sample(hp.batch_size)
    batch = Transition(*zip(*transitions))

    state_action_values = policy_net(torch.cat(batch.state, dim=1).T)[0].gather(1, torch.cat(batch.action).unsqueeze(1))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=hp.device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None], dim=1)

    next_state_values = torch.zeros(hp.batch_size, device=hp.device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states.T)[0].max(1)[0]

    expected_state_action_values = ((next_state_values * hp.gamma) + torch.cat(batch.reward)).unsqueeze(1)

    td_error = torch.abs(torch.subtract(expected_state_action_values, state_action_values))
    replay_memory.update_priorities(idx_choice, td_error)

    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 1)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
    optimizer.step()
    # scheduler.step(loss)

    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * hp.tau + \
                                     target_net_state_dict[key] * (1 - hp.tau)
    target_net.load_state_dict(target_net_state_dict)

    return loss, target_net, policy_net
