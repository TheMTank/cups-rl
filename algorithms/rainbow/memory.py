"""
Adapted from https://github.com/Kaixhin/Rainbow
"""
import random
from collections import namedtuple
import torch
import numpy as np


# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree:
    """
    We use this structure because it has the following benefits:
    - Insertion and update is O(1)
    - Sampling according to priority order is O(log n)

    Note that in practice we use a stratified sampling by dividing the memory into equal size
    segments and sampling uniformly from each one of them at every step. This reduces the sampling
    operation to O(1) as well.
    # TODO: add ascii art to explain it (propagation and stuff)
    Efficiently store millions of transitions and sample from them
    """
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        # Initialise fixed size tree with all (priority) zeros
        self.sum_tree = np.zeros((2 * size - 1, ), dtype=np.float32)
        self.data = np.array([None] * size)  # Wrap-around cyclic buffer
        self.max = 1  # Initial max value to return (1 = 1^ω)

    # Propagates value up tree given a tree index
    def _propagate(self, index, value):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate(parent, value)

    # Updates value given a tree index
    def update(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate(index, value)  # Propagate value
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data  # Store data in underlying data structure
        self.update(self.index + self.size - 1, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of a value in sum tree
    def _retrieve(self, index, value):
        """
        Binary search tree which finds index with closest probability
        """
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.sum_tree):
            return index
        elif value <= self.sum_tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.sum_tree[left])

    # Searches for a value in sum tree and returns value, data index and tree index
    def find(self, value):
        index = self._retrieve(0, value)  # Search for index of item from root
        data_index = index - self.size + 1
        return self.sum_tree[index], data_index, index  # Return value, data index, tree index

    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]


class ReplayMemory:
    def __init__(self, args, capacity):
        self.Transition = namedtuple('Transition',
                                ('timestep', 'state', 'action', 'reward', 'nonterminal'))
        self.blank_trans = self.Transition(0, torch.zeros(args.img_channels, args.resolution[0],
                                                          args.resolution[1], dtype=torch.uint8),
                                           None, 0, False)
        self.device = args.device
        self.capacity = capacity
        self.history = args.history_length
        self.discount = args.discount
        self.multi_step = args.multi_step
        # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_weight = args.priority_weight
        # Priority exponent α
        self.priority_exponent = args.priority_exponent
        self.t = 0  # Internal episode timestep counter
        # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities
        self.transitions = SegmentTree(capacity)
        self.channels = args.img_channels

    # Adds state and action at time t, reward and terminal at time t + 1
    def append(self, state, action, reward, terminal):
        state = state[-self.channels:, ...].mul(255).to(dtype=torch.uint8,
                                                        device=torch.device('cpu'))
        state = state if len(state.shape) == 3 else state.unsqueeze(0)
        # Only store last frame and discretise to save memory
        self.transitions.append(self.Transition(self.t, state, action, reward, not terminal),
                                self.transitions.max)  # Store new transition with maximum priority
        self.t = 0 if terminal else self.t + 1  # Start new episodes with t = 0

    # Returns a transition with blank states where appropriate
    def _get_transition(self, idx):
        """
        Return the idx-th transition in the SegmentTree memory information for multi-step DQN. This
        includes the transitions from t - self.history  to t + self.multi_step (multi-steps). If a
        terminal state is reached on the process or there are no previous states to time t, the
        missing transitions will be filled with blank transitions as defined in self.blank_trans
        """
        transition = np.array([None] * (self.history + self.multi_step))
        # idx is the last transition in history
        transition[self.history - 1] = self.transitions.get(idx)
        # fill in previous transitions of history
        for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
            # check if next transition is terminal/first step
            if transition[t + 1].timestep == 0:
                transition[t] = self.blank_trans  # If future frame has timestep 0
            else:
                transition[t] = self.transitions.get(idx - self.history + 1 + t)
        # fill in the history of the future for multi-step transitions
        # TODO: fix this. overlapping (sometimes), we are using the present + 3 extra frames and that final frame is considered to be the state index
        # 0 4 8 12 16 20 24 28 32 36 40
        for t in range(self.history, self.history + self.multi_step):  # e.g. 4 5 6
            if transition[t - 1].nonterminal:
                transition[t] = self.transitions.get(idx - self.history + 1 + t)
            else:
                transition[t] = self.blank_trans  # If prev/next frame is terminal
        return transition

    # Returns a valid sample from a segment
    def _get_sample_from_segment(self, segment_prob, i):
        """
        Splitting the memory in segments of segment_size, we sample uniformly from the ith segment
        and return the following information:

        prob: Transition priority (unnormalized probability)
        idx: Index of the transition sorted by time-step
        tree_idx: Index of the transition sorted by priority (index within the SegmentTree)
        (state, action, R, next_state): Transition information
        nonterminal: 1 if the episode doesn't end after or during the n-multi-steps
        TODO: ascii art

          /
         /
        /____
        """
        valid = False
        while not valid:
            # Uniformly sample an element from within a segment
            sample = np.random.uniform(i * segment_prob, (i + 1) * segment_prob)
            # Retrieve sample from tree with un-normalised probability
            prob, idx, tree_idx = self.transitions.find(sample) # TODO: explain that here is where the magic happens
            # TODO: Change straddled to something more understandable
            # Resample if transition straddled current index or probability 0
            if (self.transitions.index - idx) % self.capacity > self.multi_step and \
                    (idx - self.transitions.index) % self.capacity >= self.history and prob != 0:
                # Note that conditions are valid but extra conservative around buffer index 0
                valid = True

        # Retrieve all required transition data (from t - history_length to t + multi_step)
        transition = self._get_transition(idx)
        # Create un-discretised (float) state and nth next state
        state = torch.cat([trans.state for trans in transition[:self.history]], 0).to(
            dtype=torch.float32, device=self.device).div_(255)
        next_state = torch.cat(
            [trans.state for trans in transition[self.multi_step: (self.multi_step + self.history)]], 0).to(
            dtype=torch.float32, device=self.device).div_(255)
        # Discrete action to be used as index
        action = torch.tensor([transition[self.history - 1].action], dtype=torch.int64,
                              device=self.device)
        # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1
        # (note that invalid nth next states have reward 0)
        R = torch.tensor([sum(self.discount ** n * transition[self.history + n - 1].reward
                              for n in range(self.multi_step))], dtype=torch.float32, device=self.device)
        # Mask for non-terminal nth next states
        nonterminal = torch.tensor([transition[self.history + self.multi_step - 1].nonterminal],
                                   dtype=torch.float32, device=self.device)

        return prob, idx, tree_idx, state, action, R, next_state, nonterminal

    def sample(self, batch_size):
        """
        To sample batch_size transitions, the range [0, p_total] is divided equally into batch_size
        ranges. Next, a value is uniformly sampled from each range. Finally the transitions that
        correspond to each of these sampled values are retrieved from the sum-tree.
        Note that for efficiency reasons sampling is not done according to the priorities, i.e.
        prioritizing transitions with a higher prediction error which we can learn more from.
        """
        # Retrieve sum of all priorities (used to create a normalised probability distribution)
        p_total = self.transitions.total()
        # Batch size number of segments, based on sum over all probabilities
        segment_prob = p_total / batch_size
        # Get batch of valid consecutive samples.
        batch = [self._get_sample_from_segment(segment_prob, i) for i in range(batch_size)]
        probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)
        states, next_states, = torch.stack(states), torch.stack(next_states)
        actions, returns, nonterminals = torch.cat(actions), torch.cat(returns), \
                                         torch.stack(nonterminals)
        # Calculate normalised probabilities
        probs = np.array(probs, dtype=np.float32) / p_total
        capacity = self.capacity if self.transitions.full else self.transitions.index
        # Compute importance-sampling normalized weights w_j = w_i / max(w_i)
        # where w_i = (N * P(j))^−β)
        weights = (capacity * probs) ** -self.priority_weight
        # Normalise by max importance-sampling weight from batch
        # w_j = w_i / max(w_i)
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)
        return tree_idxs, states, actions, returns, next_states, nonterminals, weights

    def update_priorities(self, idxs, priorities):
        """
        Original formula for priorities P(i) = p_i ** α / sum_k(p_k ** α).
        We can ignore the constant sum of all elements since we are interested in retrieving the
        highest/lowest values
        """
        priorities = np.power(priorities, self.priority_exponent)
        [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]

    # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self

    # Return valid states for validation
    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration
        # Create stack of states
        state_stack = [None] * self.history
        state_stack[-1] = self.transitions.data[self.current_idx].state
        prev_timestep = self.transitions.data[self.current_idx].timestep
        # check in the whole history
        for t in reversed(range(self.history - 1)):  # e.g. 2 1 0
            # TODO: why the first state is stored blank? analyze terminals and starting point for frame stacks and multi-steps
            # checking if t+1 is a terminal state indicated by the timestep being 0
            # If a terminal state is found in the past, all the rest of past states will be blank as well
            if prev_timestep == 0:
                state_stack[t] = self.blank_trans.state  # If future frame has timestep 0
            else:
                state_stack[t] = \
                  self.transitions.data[self.current_idx + t - self.history + 1].state
                prev_timestep -= 1
        # Agent will turn into batch
        state = torch.stack(state_stack, 0).to(dtype=torch.float32, device=self.device).div_(255)
        self.current_idx += 1
        return state
