"""
Adapted from: https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py

Main A3C model which outputs predicted value, action logits and hidden state.
Includes helper functions too for weight initialisation and dynamically computing LSTM/flatten input
size.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_lstm_input_size_after_4_conv_layers(frame_dim, stride=2, kernel_size=3, padding=1,
                                     num_filters=32):
    """
    Assumes square resolution image. Find LSTM size after 4 conv layers below in A3C using regular
    Convolution math. For example:
    42x42 -> (42 − 3 + 2)÷ 2 + 1 = 21x21 after 1 layer
    11x11 after 2 layers -> 6x6 after 3 -> and finally 3x3 after 4 layers
    Therefore lstm input size after flattening would be (3 * 3 * num_filters)
    """

    width = (frame_dim - kernel_size + 2 * padding) // stride + 1
    width = (width - kernel_size + 2 * padding) // stride + 1
    width = (width - kernel_size + 2 * padding) // stride + 1
    width = (width - kernel_size + 2 * padding) // stride + 1

    return width * width * num_filters

def normalized_columns_initializer(weights, std=1.0):
    """
    Weights are normalized over their column. Also, allows control over std which is useful for
    initialising action logit output so that all actions have similar likelihood
    """

    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):
    """
    Mainly Ikostrikov's implementation of A3C (https://arxiv.org/abs/1602.01783).

    Processes an input image (with num_input_channels) with 4 conv layers,
    interspersed with 4 elu activation functions. The output of the final layer is then flattened
    and passed to an LSTM (with previous or initial hidden and cell states (hx and cx)).
    The new hidden state is used as an input to the critic and value nn.Linear layer heads,
    The final output is then predicted value, action logits, hx and cx.
    """

    def __init__(self, num_input_channels, num_outputs, frame_dim):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_input_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        # assumes square image
        self.lstm_cell_size = calculate_lstm_input_size_after_4_conv_layers(frame_dim)

        self.lstm = nn.LSTMCell(self.lstm_cell_size, 256)  # for any square input

        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
                                            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
                                            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, self.lstm_cell_size)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)

class A3C_LSTM_GA(torch.nn.Module):

    def __init__(self, num_input_channels, num_outputs, frame_dim):
        super(A3C_LSTM_GA, self).__init__()

        # Image Processing
        self.conv1 = nn.Conv2d(num_input_channels, 128, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)

        # Instruction Processing
        self.gru_hidden_size = 256
        # self.input_size = args.input_size # todo
        # self.input_size = 4 # todo change, currently "go to the microwave" 4 words
        # self.input_size = 5 # #'Turn left 3 times', 'Turn right 3 times'
        # self.input_size = 6  #['Go and look at microwave', 'Go and look at cup']
        self.input_size = 2  # ['microwave', 'cup']
        self.embedding = nn.Embedding(self.input_size, 32)
        self.gru = nn.GRU(32, self.gru_hidden_size)

        # Gated-Attention layers
        self.attn_linear = nn.Linear(self.gru_hidden_size, 64)

        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(
                # args.max_episode_length+1, # todo feed in episode length
                1000 + 1,
                self.time_emb_dim)

        # A3C-LSTM layers
        # self.linear = nn.Linear(64 * 8 * 17, 256)
        self.linear = nn.Linear(64 * 6 * 6, 256) # todo maybe 150, 150 res
        self.lstm = nn.LSTMCell(256, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, num_outputs)

        # Initializing weights
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()

    def forward(self, inputs):
        x, input_inst, (tx, hx, cx) = inputs

        # Get the image representation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_image_rep = F.relu(self.conv3(x))

        # Get the instruction representation
        encoder_hidden = torch.zeros(1, 1, self.gru_hidden_size)
        for i in range(input_inst.data.size(1)):
            # todo first unsqueeze should be sequence length!!!
            # but for now im gonna try just one word. todo check when i bring multi word sentences back
            word_embedding = self.embedding(input_inst[0, i]).unsqueeze(0).unsqueeze(0)  # crazy todo
            word_embedding = word_embedding.expand(input_inst.data.size(1), -1, -1) # only change seq len dimension
            _, encoder_hidden = self.gru(word_embedding, encoder_hidden) # todo import pdb;pdb.set_trace(). see if this works nad two unsqueezes were wrong?
        x_instr_rep = encoder_hidden.view(encoder_hidden.size(1), -1)
        # todo try MLP above instead or just one layer

        # Get the attention vector from the instruction representation
        x_attention = F.sigmoid(self.attn_linear(x_instr_rep))

        # Gated-Attention
        x_attention = x_attention.unsqueeze(2).unsqueeze(3)
        # x_attention = x_attention.expand(1, 64, 8, 17)
        x_attention = x_attention.expand(1, 64, 6, 6)
        assert x_image_rep.size() == x_attention.size()
        x = x_image_rep*x_attention
        x = x.view(x.size(0), -1)

        # A3C-LSTM
        x = F.relu(self.linear(x))
        hx, cx = self.lstm(x, (hx, cx))
        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
