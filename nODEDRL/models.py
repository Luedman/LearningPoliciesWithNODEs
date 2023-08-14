import torch
#from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint


class nODENet(torch.nn.Module):
    def __init__(self, n_observations, n_actions, no_nodes, device, no_dsteps):
        super(nODENet, self).__init__()
        self.n_observation = n_observations
        self.n_actions = n_actions
        self.ode_module = nODEUnit(n_observations, no_nodes, device)
        self.device = device
        self.no_dsteps = no_dsteps

        self.linear_out = torch.nn.Linear(n_observations, n_actions, device=device)
        self.model_type = "nODE"

        self.model_label = f"{self.model_type}_{n_observations}_{no_nodes}_{n_actions}_{device}"

    def forward(self, state):
        # inner_state = odeint(self.ode_module, state.flatten()[:, None].T,
        #                     torch.linspace(0, 1, 10, device=self.device))
        inner_state = odeint(self.ode_module, state, torch.linspace(0, 1, self.no_dsteps, device=self.device),
                             atol=0.1)
                             #rtol=0.01, atol=0.01)
        net_out = self.linear_out(inner_state[-1])
        return net_out


class nODEUnit(torch.nn.Module):
    def __init__(self, n_observation, no_nodes, device):
        super(nODEUnit, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_observation, no_nodes, device=device),
            torch.nn.Tanh(),
            torch.nn.Linear(no_nodes, no_nodes, device=device),
            torch.nn.Tanh(),
            torch.nn.Linear(no_nodes, n_observation, device=device))

        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.1)
                torch.nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        net_output = self.net(y)
        return net_output


class DeepQNet(torch.nn.Module):
    def __init__(self, n_observations, n_actions, no_nodes, device, **kwargs):
        super(DeepQNet, self).__init__()
        self.layer1 = torch.nn.Linear(n_observations, no_nodes, device=device)
        self.layer2 = torch.nn.Linear(no_nodes, no_nodes, device=device)
        self.layer3 = torch.nn.Linear(no_nodes, n_actions, device=device)
        self.model_type = "DeepQNet"

        self.model_label = f"{self.model_type}_{n_observations}_{no_nodes}_{n_actions}_{device}"

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = self.layer3(x)
        return x
