"""
Adapted from https://github.com/Kaixhin/Rainbow

Wrapper class including setup, training and evaluation functions for Rainbow DQN model
"""

import os
import numpy as np
import torch
from torch import optim

from algorithms.rainbow.model import RainbowDQN


class Agent:
    """
    Wraps control between both online and target network for setup, training and evaluation
    """
    def __init__(self, args, env):
        """
        Q(s,a) is the expected reward. Z is the full distribution from which Q is generated.
        Support represents the support of Z distribution (non-zero part of pdf).
        Z is represented with a fixed number of "atoms", which are pairs of values (x_i, p_i)
        composed by the discrete positions (x_i) equidistant along its support defined between
        Vmin-Vmax and the probability mass or "weight" (p_i) for that particular position.

        As an example, for a given (s,a) pair, we can represent Z(s,a) with 8 atoms as follows:

                   .        .     .
                .  |     .  |  .  |
                |  |  .  |  |  |  |  .
                |  |  |  |  |  |  |  |
           Vmin ----------------------- Vmax
        """
        self.action_space = env.action_space
        self.num_atoms = args.num_atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.num_atoms).to(device=args.device)
        self.delta_z = (args.V_max - args.V_min) / (self.num_atoms - 1)
        self.batch_size = args.batch_size
        self.multi_step = args.multi_step
        self.discount = args.discount

        self.online_net = RainbowDQN(args, self.action_space).to(device=args.device)
        if args.model_path and os.path.isfile(args.model_path):
            """
            When you call torch.load() on a file which contains GPU tensors, those tensors will be 
            loaded to GPU by default. You can call torch.load(.., map_location=’cpu’) and then 
            load_state_dict() to avoid GPU RAM surge when loading a model checkpoint.
            Source: https://pytorch.org/docs/stable/torch.html#torch.load
            """
            self.online_net.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        self.online_net.train()

        self.target_net = RainbowDQN(args, self.action_space).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.lr, eps=args.adam_eps)

    def reset_noise(self):
        """Resets noisy weights in all linear layers (of online net only) """
        self.online_net.reset_noise()

    def act(self, state):
        """Acts based on single state (no batch) """
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

    def act_e_greedy(self, state, epsilon=0.001):
        """
        Acts with an ε-greedy policy (used for evaluation only)
        High ε can reduce evaluation scores drastically
        """
        return np.random.randint(0, self.action_space.n) if np.random.random() < epsilon \
            else self.act(state)

    def learn(self, mem):
        """
        Executes 1 gradient descent step sampling batch_size transitions from the memory
        """
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = \
          mem.sample(self.batch_size)

        """Calculate current state probabilities (online network noise already sampled)
        The log is used to calculate the losses. It also provides more stability for the gradients 
        propagation during training and it is not needed for evaluation 
        """
        # Log probabilities log p(s_t, ·; θonline) for the visited states in the sampled transitions
        online_log_probs = self.online_net(states, log=True)
        # log p(s_t, a_t; θonline) of the actions selected on the visited states (online network)
        online_log_probs = online_log_probs[range(self.batch_size), actions]

        target_probs = self.compute_target_probs(states, actions, returns, next_states,
                                                 nonterminals)
        """Cross-entropy loss (minimises KL-distance between online and target_probs): 
        DKL(target_probs || online_probs)
        online_log_probs: policy distribution for online network
        target_probs: aligned target policy distribution
        """
        loss = -torch.sum(target_probs * online_log_probs, 1)
        self.online_net.zero_grad()
        # Backpropagate importance-weighted (Prioritized Experience Replay) minibatch loss
        (weights * loss).mean().backward()
        self.optimiser.step()
        # Update priorities of sampled transitions
        mem.update_priorities(idxs, loss.detach().cpu().numpy())

    def compute_target_probs(self, states, actions, returns, next_states, nonterminals):
        """
        Returns probability distribution for target policy given the visited transitions. Since the
        Q function is defined as a discrete distribution, the expected returns will most likely
        fall outside the support of the distribution and we won't be able to compute the KL
        divergence between the target and online policies for the visited transitions. Therefore, we
        need to project the resulting distribution into the support defined by the network output
        definition.

        For a detailed explanation of the math behind this process we recommend you to read this
        blog: https://mtomassoli.github.io/2017/12/08/distributional_rl/
        """
        with torch.no_grad():
            # Calculate self.multi_step-th next state Q distribution (Z) for Double Q-Learning
            online_z = self.online_net(next_states)
            # We compute the expectation of the Q distribution from the N-step distribution
            # online q (not distributional) = sum(z_action * p_action) for ALL actions
            online_q = (self.support.expand_as(online_z) * online_z).sum(2)
            # Store optimal action a* indices from online Q distribution
            online_greedy_action_indices = online_q.argmax(1)
            # Sample new target net noise, i.e. fix new random weights for noisy layers to
            # encourage exploration
            self.target_net.reset_noise()
            # We compute the Q distribution from the target network
            target_z = self.target_net(next_states)
            """Calculate target action probabilities for the actions selected using the online 
            network. The expected online_q will be optimized towards these values similarly as to 
            how it is done in Double DQN.
            """
            target_probs = target_z[range(self.batch_size), online_greedy_action_indices]
            """Apply distributional N-step Bellman operator Tz (Bellman operator T applied to z), 
            also Bellman equation for distributional Q.
            Tz = returns_t + γ * z_t+1 
            This is the same as the "classic" Bellman equation but using z instead of V. It
            accounts terminal states, in which case the z is 0 since we don't expect to get more 
            rewards in the future. 
            Since we are doing multi step Q-Learning as well, we will be doing a lookahead of
            self.multi_step steps. This results in the Tz operator defined as:
            Tz = returns_t + γ * R_t+1 + ... + (γ ** (n-1)) * R_t+n-1 + (γ ** n) * z_t+n
            Look at in _get_sample_from_segment() from memory.py for more details on the multi-step
            calculations
            For calculating Tz we use the support to calculate ALL possible expected returns, 
            i.e. the full distribution, without looking at the probabilities yet. 
            """
            Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.multi_step) \
                 * self.support.unsqueeze(0)
            # Clamp values so they fall within the support of Z values
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
            """Compute L2 projection of Tz onto fixed support Z in two steps.
            1. Find which values of the discrete fixed distribution are the closest lower (l) and 
            upper value (u) to the values obtained from Tz (b). As a reminder, b is the new support 
            of our return distribution shifted from the original network output support when we 
            computed Tz. In other words, b is how many times deltaz I am from Vmin to get to 
            Tz by definition
            b = (Tz - Vmin) / Δz 
            We've expressed Tz in terms of b (the misaligned support but still similar in the sense 
            of exact same starting point and exact same distance between the atoms as the original 
            support). Still misaligned but that's why do the redistribution in terms of 
            proportionality.
            """
            b = (Tz - self.Vmin) / self.delta_z
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            """
            2. Distribute probability of Tz. Since b is most likely not having the exact value of 
            one of our predefined atoms, we split its probability mass between the closest atoms 
            (l, u) in proportion to their OPPOSED distance to b so that the closest atom receives
            most of the mass in proportion to their distances.
                                  u
                      l    b      .     
                      ._d__.__2d__|    
                 ...  |    :      |  ...    mass_l += mass_b * 2 / 3
                      |    :      |         mass_u += mass_b * 1 / 3
            Vmin ----------------------- Vmax

            The probability mass becomes 0 when l = b = u (b is int). Note that for this case
            u - b + b - l = b - b + b - b = 0 
            To fix this, we change  l -= 1 which would result in:
            u - b + b - l = b - b + b - (b - 1) = 1
            Except in the case where b = 0, because l -=1 would make l = -1  
            Which would mean that we are subtracting the probability mass! To handle this case we
            would only do l -=1 if u > 0, and for the particular case of b = u = l = 0 we would 
            keep l = 0 but u =+ 1
            """
            l[(u > 0) * (l == u)] -= 1  # Handles the case of u = b = l != 0
            u[(l < (self.num_atoms - 1)) * (l == u)] += 1  # Handles the case of u = b = l = 0

            # We use new_zeros instead of zeros to auto assign device and dtype of states tensor
            projected_target_probs = states.new_zeros(self.batch_size, self.num_atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.num_atoms),
                                    self.batch_size).unsqueeze(1).expand(self.batch_size,
                                                                         self.num_atoms).to(actions)
            """Distribute probabilities to the closest lower atom in inverse proportion to the
            distance to the atom. For efficiency, we are adding the values to the flattened view of 
            the array not flattening the array itself.
            """
            projected_target_probs.view(-1).index_add_(
                0, (l + offset).view(-1), (target_probs * (u.float() - b)).view(-1)
            )
            # Add probabilities to the closest upper atom
            projected_target_probs.view(-1).index_add_(
                0, (u + offset).view(-1), (target_probs * (b - l.float())).view(-1)
            )
        return projected_target_probs

    def update_target_net(self):
        """Updates target network as explained in Double DQN """
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path, filename):
        """Save model parameters on current device """
        torch.save(self.online_net.state_dict(), os.path.join(path, filename))

    def evaluate_q(self, state):
        """Evaluates Q-value based on single state (no batch) """
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
