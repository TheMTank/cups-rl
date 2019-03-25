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
        self.num_atoms = args.num_atoms
        self.action_space = action_space.n
        self.linear_in = self.get_linear_size(args.resolution)
        self.conv1 = nn.Conv2d(args.img_channels * args.history_length, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        # Fully connected hidden features to value stream (fc_h_v) and advantage stream (fc_h_a)
        self.fc_h_v = NoisyLinear(self.linear_in, args.hidden_size, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(self.linear_in, args.hidden_size, std_init=args.noisy_std)
        # Fully connected output to generate value (fc_z_v) and advantage (fc_z_a) distributions
        self.fc_z_v = NoisyLinear(args.hidden_size, self.num_atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(args.hidden_size, self.action_space * self.num_atoms,
                                  std_init=args.noisy_std)

    def forward(self, x, log=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.linear_in)
        # the "z_" prefix is used here to indicate that the value is defined as a distribution
        # instead of a single value. Check "agent.py" for more detailed information.
        z_v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        z_a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
        z_v, z_a = z_v.view(-1, 1, self.num_atoms), z_a.view(-1, self.action_space, self.num_atoms)
        z_q = z_v + z_a - z_a.mean(1, keepdim=True)  # Combine streams
        # log softmax used while learning to generate probabilities with higher numerical stability
        if log:
            z_q = F.log_softmax(z_q, dim=2)  # Log probabilities with action over second dimension
        else:
            z_q = F.softmax(z_q, dim=2)  # Probabilities with action over second dimension
        # distributional Q shape: batch_size x num_actions x num_atoms
        return z_q

    def reset_noise(self):
        for name, module in self.named_children():
          if 'fc' in name:
            module.reset_noise()

    @staticmethod
    def get_linear_size(resolution):
        """
        Calculates the size of the input features for the Linear layers depending on the input
        resolution given to the CNN.

        The output of the convolution of 1 feature map with a kernel is calculated as follows
        (for each dimension):
            out_dim_size = ((input_size − kernel_size + 2 * padding) // stride) + 1
        The input size to the linear layer is "flattened" to a 1 dimensional vector with size:
            in_size = h * w * n_feature_maps
        """
        linear_size = 64  # number of filters before linear size
        for dim in resolution:
            out_conv1 = ((dim - 8 + 2) // 4) + 1
            out_conv2 = ((out_conv1 - 4 + 0) // 2) + 1
            out_conv3 = (out_conv2 - 3 + 0) + 1
            linear_size *= out_conv3  # multiply the number of filters by width and height
        return linear_size


class NoisyLinear(nn.Module):
    """
    From the paper "Noisy Networks for exploration"
    Source: https://arxiv.org/pdf/1706.10295.pdf

    Factorised NoisyLinear layer with bias, which uses an independent noise per each output and
    another independent noise per each input.

    This layer replaces a "linear" layer for one that describes the weights with a distribution
    made of learnable parameters (mu, sigma) and a non learnable random factor (epsilon).
    According to the paper, can be used to replace e-greedy exploration.

    Normal Linear: outputs = weights * in_features + bias
    Noisy: outputs = (µ_weights + σ_weights * ε_weights) * in_features + µ_bias + σ_bias * ε_bias
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        """ This is typically used to register a buffer that should not to be considered a 
        model parameter. For example, BatchNorm’s running_mean is not a parameter, but is part of 
        the persistent state.
        Source:  https://pytorch.org/docs/stable/nn.html#torch.nn.Module.register_buffer """
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """
        Learnable noise initialization.
        µ is sampled from a uniform distribution U[− √1/size, + √1/size]
        σ is initialized to std_init/size
        """
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size):
        """ Sample values to compute random generation factor to scale sigmas f(x) = sgn(x)p|x|"""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """
        Factorised Gaussian noise is used to reduce computation time of random number generation
        instead of Independent Gaussian noise. The number of sampled parameters for comparison are:
        Factorised Gaussian noise: in_features + out_features
        Independent Gaussian noise: in_features * out_features + out_features
        The factors are defined as follows:
        ε_weights_i,j = f(εi)f(εj)
        ε_biases_j = f(εj)
        f(x) = sgn(x) * sqrt(|x|)
        """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))  # outer product e_j x e_i
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, layer_input):
        # PyTorch nn.Module have an attribute self.training to indicate training or evaluation mode
        # You can switch between training and evaluation with dqn.train() or dqn.eval()
        if self.training:
            return F.linear(layer_input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(layer_input, self.weight_mu, self.bias_mu)
