import os

import gym
import numpy as np
import torch
from matplotlib import cm, colors
from matplotlib import pyplot as plt

from nODEDRL.models import nODENet
from nODEDRL.modules import ReplayMemory, HyperParameterWrapper, Transition


def draw_state(model_file_name):
    # model_file_name = "nODENet_4_128_2_cpu_lr0.001-y0.999-t0.01-policy-net.pth"
    (model_type, n_observations, no_nodes, n_actions, device, _) = model_file_name.split('_')
    path = os.path.join(os.getcwd(), "models", model_file_name)
    model = nODENet(int(n_observations), int(n_actions), int(no_nodes), device, 10)
    model.load_state_dict(torch.load(path))

    env = gym.make("CartPole-v1")
    ax = plt.figure().add_subplot(projection='3d')

    pos_low = env.observation_space.low[0]
    pos_high = env.observation_space.high[0]
    angl_low = env.observation_space.low[2]
    angl_high = env.observation_space.high[2]
    norm = colors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap('hsv')

    for init_position in np.linspace(pos_low, pos_high, num=10):
        for init_angle in np.linspace(angl_low, angl_high, num=10):
            color = cmap(norm(init_position / (pos_high - pos_low)))
            color = color[:-1] + (0.5,)

            initial_state = torch.tensor([0.0, 0.0, 0.0, 0.0])
            initial_state[0] = init_position
            initial_state[2] = init_angle
            net_out, inner_state = model.forward_state(torch.tensor(initial_state))
            x = np.linspace(0.0, 1.0, 10)
            z = inner_state[:, 0].detach().numpy()
            y = inner_state[:, 2].detach().numpy()
            ax.plot(x, y, z, color=color)
    ax.set(
        zlabel='Position',
        ylabel='Angle',
        xlabel='Time',
    )
    plt.savefig(model_file_name.split(".")[0] + "inner_state.png")
    plt.show()


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
                   cmap=cm.coolwarm, label="actions played")
        ax.scatter(states_0.cpu().numpy(), states_1.cpu().numpy(),
                   torch.cat(net_actions).cpu().numpy(),
                   cmap=cm.coolwarm, label="net policy")
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
