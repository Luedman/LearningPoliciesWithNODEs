import math
import random
import time
from collections import namedtuple, deque

import matplotlib
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchdiffeq import odeint

is_ipython = 'inline' in matplotlib.get_backend()

Transition = namedtuple("Transition", ('state', 'action', 'next_state', 'reward'))


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
        self.layer1 = torch.nn.Linear(n_observations, 32, device=device)
        self.layer2 = torch.nn.Linear(32, 32, device=device)
        self.layer3 = torch.nn.Linear(32, n_actions, device=device)
        self.model_type = "DeepQNet"

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x.T))
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
                             torch.linspace(0, 1, 100, device=self.device))
        net_out = self.linear_out(inner_state[-1])
        return net_out


class nODEUnit(torch.nn.Module):
    def __init__(self, n_observation, device):
        super(nODEUnit, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_observation, 32, device=device),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32, device=device),
            torch.nn.Tanh(),
            torch.nn.Linear(32, n_observation, device=device))

        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.1)
                torch.nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        net_output = self.net(y)
        return net_output


def select_action(state, params, env, epoch, model):
    eps_threshold = params.get("eps_end") + (params.get("eps_start") - params.get("eps_end")) * \
                    math.exp(-1. * epoch / params.get("eps_decay"))
    with torch.no_grad():
        rand = random.random()
        if rand > eps_threshold and model is not None:
            action_values = model(state)
            action_idx = action_values.argmax()
            action_value = get_action_value(action_idx, params)
        else:
            action_value = torch.tensor(env.action_space.sample().reshape(1,),
                                        device=params.get("device"),
                                        dtype=torch.float)
    return action_value, eps_threshold


def get_action_index(action_value, params) -> torch.tensor:
    action_index = torch.bucketize(action_value, torch.linspace(params.get("action_low")[0],
                                                                params.get("action_high")[0],
                                                                params.get("no_adpoints"),
                                                                device=params.get("device")))
    return action_index


def get_action_value(action_idx, params) -> torch.tensor:
    action_value = torch.linspace(params.get("action_low")[0],
                                  params.get("action_high")[0],
                                  params.get("no_adpoints"),
                                  device=params.get("device"))[action_idx][None]
    return action_value


def run_model(env, model, replay_memory, params, run_label):
    writer = SummaryWriter(log_dir=f"runs/{model.model_type}_{run_label}")
    device = params.get("device")

    optimizer = torch.optim.RMSprop(model.parameters(), lr=params.get("learning_rate"))

    loss_per_epoch, reward_per_epoch, avg_loss, avg_reward = [], [], [], []
    no_solves = 0
    for epoch in range(1, params.get("no_epochs") + 1):
        start_time = time.time()
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)[:, None, None]
        total_reward_per_epoch = 0
        steps = 0
        done = False
        while not done:
            action, eps_threshold = select_action(state, params, env, epoch, model=model)

            for i in range(1):
                observation, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
                done = (terminated or truncated)

                total_reward_per_epoch += reward
                reward = torch.tensor([reward], device=device)

                if done:
                    print(f"E{epoch}: Done after {steps} steps, terminated: {terminated}, truncated: {truncated}," +
                          f" reward: {total_reward_per_epoch:.2f}, time: {time.time() - start_time:.2f} sec")
                    no_solves += int(terminated)
                    steps = 0
                    next_state = None
                    reward_per_epoch.append(total_reward_per_epoch)
                    avg_reward.append(torch.mean(torch.tensor(reward_per_epoch[-20:])))
                    writer.add_scalar('Reward/train', total_reward_per_epoch, epoch)
                    writer.add_scalar('Reward/train_avg', torch.mean(torch.tensor(reward_per_epoch[-20:])), epoch)
                    writer.add_scalar('Epsilon Threshold/train', eps_threshold, epoch)
                    writer.add_scalar('No. Solves',no_solves, epoch)
                    break
                else:
                    next_state = torch.reshape(torch.tensor(observation, dtype=torch.float32, device=device), (2, 1, 1))
                    steps += 1

                replay_memory.push(state, action, next_state, reward)
                state = next_state

        if len(replay_memory) > params.get("batch_size"):
            transitions = replay_memory.sample(params.get("batch_size"))
            batch = Transition(*zip(*transitions))

            state_action_values = []
            for state, reward, action in zip(batch.state, batch.reward, batch.action):
                action_values = model(state)
                action_idx = get_action_index(action, params)
                q_value = action_values.flatten()[action_idx]
                state_action_values.append(q_value)

            expected_state_action_values = []
            for next_state, reward in zip(batch.next_state, batch.reward):
                if next_state is not None:
                    exp_action_values = model(next_state)
                    exp_q_value_next_state = exp_action_values.max()
                    expected_state_action_values.append((exp_q_value_next_state * params.get("gamma")) + reward)
                else:
                    expected_state_action_values.append(reward)

            state_action_values = torch.cat(state_action_values)
            expected_state_action_values = torch.cat(expected_state_action_values)

            criterion = torch.nn.MSELoss()
            loss = criterion(state_action_values, expected_state_action_values)

            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 100)
            optimizer.step()

            loss_per_epoch.append(loss.item())
            avg_loss.append(torch.mean(torch.mean(torch.tensor(loss_per_epoch[-20:]))))

            writer.add_scalar('Loss/train', loss.item(), epoch)
            writer.add_scalar('Loss/train_avg', torch.mean(torch.tensor(loss_per_epoch[-20:])), epoch)
        # print(f"Total Epoch Time {time.time() - start_time:.2f} sec")

        if epoch % 25 == 0:
            print(f"E{epoch}: Loss {loss.item():.6f}, Esp: {eps_threshold:.2f}")
            generate_charts(epoch, replay_memory, model, params,
                            loss_per_epoch, avg_loss, reward_per_epoch, avg_reward)


def generate_charts(epoch, replay_memory, ode_net, params,
                    loss_per_epoch, avg_loss, reward_per_epoch, avg_reward):
    with torch.no_grad():
        transitions = replay_memory.get_memory()
        full_batch = Transition(*zip(*transitions))

        net_actions = []
        for state in full_batch.state:
            action_idx = ode_net(state).argmax()
            action_value = get_action_value(action_idx, params)
            net_actions.append(action_value)

        states_0 = torch.cat(full_batch.state, dim=1)[0, :]
        states_1 = torch.cat(full_batch.state, dim=1)[1, :]
        actions = torch.cat(full_batch.action)

        ax = plt.figure(1).add_subplot(projection='3d')
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

    if epoch != params.get("no_epchs") and False:
        plt.pause(0.0001)

    plt.savefig("training_progress.png")
    plt.clf()

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
