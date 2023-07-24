import math
import os
import random
import time
from collections import namedtuple, deque
from itertools import count
from operator import itemgetter

import matplotlib
import numpy as np
import torch
from gym.spaces import Box, Discrete
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

# from torchdiffeq import odeint
from nODEDRL.models import nODENet, DeepQNet

is_ipython = 'inline' in matplotlib.get_backend()

Transition = namedtuple("Transition", ('state', 'action', 'next_state', 'reward'))


def load_model(model_label: str):
    (model_type, n_observations, no_nodes, n_actions, device, _) = model_label.split('_')
    path = os.path.join(os.getcwd(), "models", model_label)
    if model_type == "nODE":
        model = nODENet(int(n_observations), int(n_actions), int(no_nodes), device, 10)
        model.load_state_dict(torch.load(path))
    elif model_type == "DeepQNet":
        model = DeepQNet(int(n_observations), int(n_actions), int(no_nodes), device)
        model.load_state_dict(torch.load(path))
    else:
        raise ValueError("model_type unknown")
    return model


class HyperParameterWrapper:
    def __init__(self, env,
                 epsilon_start: float,
                 epsilon_end: float,
                 learning_rate: float,
                 no_epochs: int,
                 no_dsteps: int = None,
                 no_nodes: int = 32,
                 epsilon_decay: int = None,
                 learning_mode="off-policy",
                 batch_size: int = 256,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 period_length: int = 1,
                 device_str: str = None,
                 action_dpoints: int = None,
                 label: str = None):
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.action_dpoints = action_dpoints
        self.gamma = gamma
        self.tau = tau
        self.learning_mode = learning_mode
        self.no_nodes = no_nodes
        self.no_dsteps = no_dsteps

        self.eps_start = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay or no_epochs * 2

        self.learning_rate = learning_rate
        self.no_epochs = no_epochs
        self.batch_size = batch_size

        self.period_length = period_length
        self.label = f'_{label}' if label is not None else ''

        if device_str is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        if type(env.action_space) is Box:
            self.action_type = "box"
            self.action_low, self.action_high = env.action_space.low, env.action_space.high
            self.disc_action_space = torch.linspace(self.action_low[0],
                                                    self.action_high[0],
                                                    self.action_dpoints, device=self.device)
            self.torch_action_type = torch.float

        elif type(env.action_space) is Discrete:
            self.action_type = "discrete"
            self.action_low, self.action_high = (0, env.action_space.n)
            self.disc_action_space = list(range(env.action_space.n))
            self.action_dpoints = env.action_space.n
            self.torch_action_type = torch.int64

    @property
    def short_label(self) -> str:
        return f"{self.no_nodes}_lr{str(self.learning_rate)}_y{self.gamma}_t{self.tau}" + self.label

    def epsilon_threshold(self, episode) -> float:
        if episode is None:
            eps_threshold = 0.0
        elif self.learning_mode == 'eps_decay_log':
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * episode / self.eps_decay)
        elif self.learning_mode == 'eps_decay_linear':
            eps_threshold = max([self.eps_start - (self.eps_start - self.eps_end) * (episode / self.no_epochs) * 2,
                                 self.eps_end])
        elif self.learning_mode == 'off-policy':
            if episode / self.no_epochs < 0.9:
                eps_threshold = self.eps_start
            else:
                eps_threshold = 0.0
        elif self.learning_mode == 'on-policy':
            if episode / self.no_epochs < 0.9:
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


def select_action(env,
                  state: torch.tensor,
                  epoch: int,
                  policy_net: torch.nn.Module,
                  hp: HyperParameterWrapper) -> (torch.tensor, float):
    eps_threshold = hp.epsilon_threshold(epoch)
    with torch.no_grad():
        rand = random.random()
        if rand > eps_threshold:
            action_values = policy_net(state.T)
            action_idx = action_values.argmax()
            action_value = hp.disc_action_space[action_idx]
        else:
            action_value = env.action_space.sample()
        action_value_tensor = torch.tensor(action_value,
                                           device=hp.device,
                                           dtype=hp.torch_action_type).reshape(1, )
    return action_value_tensor, eps_threshold


def make_tensorboard_writer(model_type: str, short_label: str) -> SummaryWriter:
    tensorboard_label = f'{model_type}_{short_label}'
    writer = SummaryWriter(log_dir=f"runs/{tensorboard_label}")
    return writer


def run_environment(policy_net, env, hp, epoch):
    state_transitions = []
    action_values = 0
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=hp.device)[:, None, None]
    total_reward_per_epoch = 0
    for steps in count():
        action, eps_threshold = select_action(env, state, epoch, policy_net=policy_net, hp=hp)
        action_values += action

        observation, reward, terminated, truncated, _ = env.step(hp.conv_action(action))
        done = (terminated or truncated)

        if done:
            next_state = None
            break
        else:
            next_state = torch.reshape(torch.tensor(observation,
                                                    dtype=torch.float32,
                                                    device=hp.device,
                                                    requires_grad=True),
                                       (len(hp.obs_high), 1, 1))
            steps += 1

        total_reward_per_epoch += reward
        reward = torch.tensor([reward], device=hp.device)

        transition = (state, action, next_state, reward)
        state_transitions.append(transition)
        return state_transitions, terminated, truncated, total_reward_per_epoch


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


def run_model(env,
              model_class,
              hp: HyperParameterWrapper):
    replay_memory = ReplayMemory(1_000)

    policy_net = model_class(n_observations=len(hp.obs_high),
                             n_actions=hp.action_dpoints,
                             device=hp.device,
                             no_nodes=hp.no_nodes,
                             no_dsteps=hp.no_dsteps)
    target_net = model_class(n_observations=len(hp.obs_high),
                             n_actions=hp.action_dpoints,
                             device=hp.device,
                             no_nodes=hp.no_nodes,
                             no_dsteps=hp.no_dsteps)

    writer = make_tensorboard_writer(policy_net.model_type, hp.short_label)
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=hp.learning_rate, amsgrad=True)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=10_000, min_lr=1e-6,
                                  cooldown=10_000, factor=0.5)

    loss_per_epoch, reward_per_epoch, avg_loss, avg_reward = [], [], [], []
    no_solves = 0
    total_steps = 0
    action_values = 0
    loss = 0
    start_time_total = time.time()
    for epoch in range(1, hp.no_epochs + 1):
        start_time = time.time()

        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=hp.device)[:, None, None]
        total_reward_per_epoch = 0
        for steps in count():
            action, eps_threshold = select_action(env, state, epoch, policy_net=policy_net, hp=hp)
            total_steps += 1
            action_values += action

            observation, reward, terminated, truncated, _ = env.step(hp.conv_action(action))
            done = (terminated or truncated)

            if truncated:
                no_solves += 1
                next_state = None
            elif terminated:
                next_state = None
            else:
                next_state = torch.reshape(torch.tensor(observation,
                                                        dtype=torch.float32,
                                                        device=hp.device,
                                                        requires_grad=True),
                                           (len(hp.obs_high), 1, 1))
                steps += 1

            total_reward_per_epoch += reward
            reward = torch.tensor([reward], device=hp.device)

            replay_memory.push(state, action, next_state, reward)

            state = next_state

            if len(replay_memory) > hp.batch_size:
                loss, target_net, policy_net = optimize_model(replay_memory, hp, policy_net, target_net,
                                                              optimizer, scheduler)

                loss_per_epoch.append(loss.item())
                avg_loss.append(torch.mean(torch.mean(torch.tensor(loss_per_epoch[-20:]))))

                writer.add_scalar('Loss/train', loss.item(), total_steps)
                writer.add_scalar('Loss/train_avg', torch.mean(torch.tensor(loss_per_epoch[-200:])), total_steps)

            if done:
                reward_per_epoch.append(total_reward_per_epoch)
                avg_reward.append(torch.mean(torch.tensor(reward_per_epoch[-20:])))
                writer.add_scalar('Reward/train', total_reward_per_epoch, epoch)
                writer.add_scalar('Reward/train_avg', torch.mean(torch.tensor(reward_per_epoch[-20:])), epoch)
                writer.add_scalar('Epsilon Threshold/train', eps_threshold, epoch)
                writer.add_scalar('No. Solves', no_solves, epoch)
                writer.add_scalar('Steps', steps, epoch)
                writer.add_scalar('Avg. Action value', action_values / total_steps, epoch)
                writer.add_scalar('Time/Epoch', time.time() - start_time, epoch)
                writer.add_scalar('Time/Total', time.time() - start_time_total, epoch)
                if epoch % 25 == 0:
                    print(f"E{epoch}: Done after {steps} steps, terminated: {terminated}, truncated: {truncated}," +
                          f" reward: {total_reward_per_epoch:.2f}, time: {time.time() - start_time:.2f} sec,"
                          f" eps: {eps_threshold:.2f}, loss {loss:.2f}, {hp.short_label}")
                break

            # generate_charts(epoch, replay_memory, policy_net, hp, loss_per_epoch, avg_loss,
            #                reward_per_epoch, avg_reward)

    model_label_tn = f"{target_net.model_label}_target-net.pth"
    model_label_pn = f"{target_net.model_label}_policy-net.pth"
    torch.save(target_net.state_dict(), os.path.join(os.getcwd(), "models", model_label_tn))
    torch.save(policy_net.state_dict(), os.path.join(os.getcwd(), "models", model_label_pn))
    print(f'{model_label_tn} saved')
    print(f'{model_label_pn} saved')


def optimize_model(replay_memory: ReplayMemory,
                   hp: HyperParameterWrapper,
                   policy_net: torch.nn.Module,
                   target_net: torch.nn.Module,
                   optimizer: torch.optim,
                   scheduler: torch.optim):
    idx_choice, transitions = replay_memory.sample(hp.batch_size)
    batch = Transition(*zip(*transitions))

    state_action_values = (torch.cat(batch.state, dim=1).T)[0].gather(1, torch.cat(batch.action).unsqueeze(1))

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


def generate_charts(epoch: int,
                    replay_memory: ReplayMemory,
                    policy_net: torch.nn.Module,
                    hp: HyperParameterWrapper,
                    loss_per_epoch: list,
                    avg_loss: list,
                    reward_per_epoch: list,
                    avg_reward: list):
    with torch.no_grad():
        transitions = replay_memory.get_memory()
        full_batch = Transition(*zip(*transitions))

        net_actions = []
        for state in full_batch.state:
            action_idx = policy_net(state.T).argmax()
            action_value = hp.disc_action_space[action_idx]
            net_actions.append(torch.tensor(hp.conv_action(action_value)).reshape(1, ))

        states_0 = torch.cat(full_batch.state, dim=1)[0, :]
        states_1 = torch.cat(full_batch.state, dim=1)[1, :]
        actions = torch.cat(full_batch.action)

        fig = plt.figure(1)
        ax = fig.add_subplot(projection='3d')
        ax.scatter(states_0.cpu().numpy(), states_1.cpu().numpy(), actions.cpu().numpy(),
                   cmap=matplotlib.cm.coolwarm, label="actions played")
        ax.scatter(states_0.cpu().numpy(), states_1.cpu().numpy(),
                   torch.cat(net_actions).cpu().numpy(),
                   cmap=matplotlib.cm.coolwarm, label="net policy")
        ax.set_xlabel("state 0: position")
        ax.set_ylabel("state 1: velocity")
        ax.set_zlabel("action")
        plt.legend()
        plt.title(epoch)

        plt.savefig("training_progress.png")
        plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(loss_per_epoch, label="Loss per epoch")
    ax1.plot(avg_loss, label="20 moving average loss")
    ax1.set_title("Training Loss")
    ax1.legend()

    ax2.plot(reward_per_epoch, label="Reward per epoch")
    ax2.plot(avg_reward, label="20 moving average reward")
    ax2.set_title("Training Reward")
    ax2.legend()

    fig.savefig("training_loss.png")
    plt.close(fig)
