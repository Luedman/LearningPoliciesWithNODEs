import math
import random
import time
from collections import namedtuple, deque
from itertools import count

import matplotlib
import torch
from gym.spaces import Box, Discrete
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchdiffeq import odeint

is_ipython = 'inline' in matplotlib.get_backend()

Transition = namedtuple("Transition", ('state', 'action', 'next_state', 'reward'))


class HyperParameterWrapper:
    def __init__(self, env,
                 action_dpoints: int,
                 gamma: float,
                 tau: float,
                 epsilon_start: float,
                 epsilon_end: float,
                 epsilon_decay: int,
                 learning_rate: float,
                 no_epochs: int,
                 batch_size: int,
                 period_length: int = 1,
                 device_str: str = None,
                 label: str = None):
        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        self.action_dpoints = action_dpoints
        self.gamma = gamma
        self.tau = tau

        self.eps_start = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay

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
        return f"adp{self.action_dpoints}_g{self.gamma}_edc{self.eps_decay}_" \
               f"pl{self.period_length}" + self.label

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


class ReplayMemory:
    def __init__(self, capacity: int):
        self._memory = deque([], maxlen=capacity)

    def __len__(self):
        return len(self._memory)

    def push(self, state, action, next_state, reward):
        self._memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self._memory, batch_size)

    def get_memory(self):
        return self._memory


class DeepQNet(torch.nn.Module):
    def __init__(self, n_observations, n_actions, device):
        super(DeepQNet, self).__init__()
        self.layer1 = torch.nn.Linear(n_observations, 128, device=device)
        self.layer2 = torch.nn.Linear(128, 128, device=device)
        self.layer3 = torch.nn.Linear(128, n_actions, device=device)
        self.model_type = "DeepQNet"

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = torch.nn.functional.relu(self.layer3(x))
        return x


class nODEnet(torch.nn.Module):
    def __init__(self, n_observations, n_actions, device, label=None):
        super(nODEnet, self).__init__()
        self.n_observation = n_observations
        self.n_actions = n_actions
        self.ode_module = nODEUnit(n_observations, device)
        self.device = device

        self.linear_out = torch.nn.Linear(n_observations, n_actions, device=device)
        self.model_type = "nODE"

    def forward(self, state):
        inner_state = odeint(self.ode_module, state.flatten()[:, None].T,
                             torch.linspace(0, 1, 10, device=self.device))
        net_out = self.linear_out(inner_state[-1])
        return net_out


class nODEUnit(torch.nn.Module):
    def __init__(self, n_observation, device):
        super(nODEUnit, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_observation, 128, device=device),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128, device=device),
            torch.nn.Tanh(),
            torch.nn.Linear(128, n_observation, device=device))

        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.1)
                torch.nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        net_output = self.net(y)
        return net_output


def select_action(state: torch.tensor,
                  hp: HyperParameterWrapper,
                  env,
                  total_steps: int,
                  policy_net: torch.nn.Module) -> (torch.tensor, float):
    eps_threshold = hp.eps_start
    # eps_threshold = 1 - (epoch / hp.no_epochs)
    eps_threshold = hp.eps_end + (hp.eps_start - hp.eps_end) * math.exp(-1. * total_steps / hp.eps_decay)
    with torch.no_grad():
        rand = random.random()
        if rand > eps_threshold:
            action_values = policy_net(state.T)
            action_idx = action_values.argmax()
            action_value = hp.disc_action_space[action_idx]
        else:
            action_value = env.action_space.sample()
        action_value_tensor = torch.tensor(action_value, device=hp.device,
                                           dtype=hp.torch_action_type).reshape(1, )
    return action_value_tensor, eps_threshold


def run_model(env,
              model_class: torch.nn.Module,
              replay_memory: ReplayMemory,
              hp: HyperParameterWrapper):
    policy_net = model_class(n_observations=len(hp.obs_high),
                             n_actions=hp.action_dpoints,
                             device=hp.device)
    target_net = model_class(n_observations=len(hp.obs_high),
                             n_actions=hp.action_dpoints,
                             device=hp.device)

    writer = SummaryWriter(log_dir=f"runs/{policy_net.model_type}_{hp.short_label}")
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=hp.learning_rate, amsgrad=True)

    loss_per_epoch, reward_per_epoch, avg_loss, avg_reward = [], [], [], []
    no_solves = 0
    total_steps = 0
    action_values = 0
    for epoch in range(1, hp.no_epochs + 1):
        start_time = time.time()
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=hp.device)[:, None, None]
        total_reward_per_epoch = 0
        for steps in count():
            action, eps_threshold = select_action(state, hp, env, total_steps, policy_net=policy_net)
            total_steps += 1
            action_values += action

            observation, reward, terminated, truncated, _ = env.step(hp.conv_action(action))
            done = (terminated or truncated)

            total_reward_per_epoch += reward
            reward = torch.tensor([reward], device=hp.device)

            if terminated:
                no_solves += int(truncated)
                next_state = None
            else:
                next_state = torch.reshape(torch.tensor(observation, dtype=torch.float32, device=hp.device),
                                           (len(hp.obs_high), 1, 1))
                steps += 1

            replay_memory.push(state, action, next_state, reward)
            state = next_state

            if len(replay_memory) > hp.batch_size:
                loss, target_net, policy_net = optimize_model(replay_memory, hp, policy_net, target_net, optimizer)

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
                writer.add_scalar('Avg. Action value', action_values/total_steps, epoch)
                print(f"E{epoch}: Done after {steps} steps, terminated: {terminated}, truncated: {truncated}," +
                      f" reward: {total_reward_per_epoch:.2f}, time: {time.time() - start_time:.2f} sec,"
                      f" eps: {eps_threshold:.2f}")
                break

        if epoch % 100 == 0 and False:
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]
            target_net.load_state_dict(target_net_state_dict)

            # generate_charts(epoch, replay_memory, policy_net, hp, loss_per_epoch, avg_loss,
            #                reward_per_epoch, avg_reward)


def optimize_model(replay_memory: ReplayMemory,
                   hp: HyperParameterWrapper,
                   policy_net: torch.nn.Module,
                   target_net: torch.nn.Module,
                   optimizer: torch.optim):
    transitions = replay_memory.sample(hp.batch_size)
    batch = Transition(*zip(*transitions))

    state_action_values = policy_net(torch.cat(batch.state, dim=1).T)[0].gather(1, torch.cat(batch.action).unsqueeze(1))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=hp.device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None], dim=1)

    next_state_values = torch.zeros(hp.batch_size, device=hp.device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states.T)[0].max(1)[0]

    expected_state_action_values = (next_state_values * hp.gamma) + torch.cat(batch.reward)

    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 1)
    optimizer.step()

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
