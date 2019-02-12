"""
Adapted from https://github.com/Kaixhin/Rainbow

Model definition definition for DQN and Noisy layers
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class RainbowDQN(nn.Module):
    """
    From the paper "Rainbow: Combining Improvements in Deep Reinforcement Learning"
    Source: https://arxiv.org/pdf/1710.02298.pdf

    Main model used as Q function estimator for policy generation
    """
    def __init__(self, args, action_space):
        super().__init__()
        self.atoms = args.atoms
        self.action_space = action_space.n
        self.linear_in = self.get_linear_size(args.resolution)
        self.conv1 = nn.Conv2d(args.in_channels, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc_h_v = NoisyLinear(self.linear_in, args.hidden_size, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(self.linear_in, args.hidden_size, std_init=args.noisy_std)
        self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(args.hidden_size, self.action_space * self.atoms, std_init=args.noisy_std)

    def forward(self, x, log=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.linear_in)
        v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams
        # TODO: add info over why do we use it and when (log)
        # TODO: change name of Q to Z when appropriate and document why we softmax and what we get
        if log:  # Use log softmax for numerical stability
          q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
        else:
          q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
        return q

    def reset_noise(self):
        for name, module in self.named_children():
          if 'fc' in name:
            module.reset_noise()

    @staticmethod
    def get_linear_size(resolution):
        """
        Calculates the size of the input features for the Linear layers
        """
        linear_size = 64  # number of filters before linear size
        for dim in resolution:
            out_conv1 = ((dim - 8 + 2) // 4) + 1
            out_conv2 = ((out_conv1 - 4) // 2) + 1
            out_conv3 = (out_conv2 - 3) + 1
            linear_size *= out_conv3
        return linear_size


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    """
    From the paper "Noisy Networks for exploration"
    Source: https://arxiv.org/pdf/1706.10295.pdf

    This layer replaces a "linear" layer for one that describes the weights with a distribution
    made of learnable parameters (mu, sigma). According to the paper, can be used to replace
    e-greedy exploration.
    # TODO: write equations for noisy layers so it is more understandable
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, layer_input):
        if self.training:
            return F.linear(layer_input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(layer_input, self.weight_mu, self.bias_mu)
