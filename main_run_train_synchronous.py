from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp

import my_optim
# from envs import create_atari_env
import envs
from model import ActorCritic
from a3c_lstm_ga_model import A3C_LSTM_GA
from test import test
from train import train
from train_a3c_lstm_ga import train_a3c_lstm_ga

# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=1,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='PongDeterministic-v4',
                    help='environment to train on (default: PongDeterministic-v4)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = "" # todo change, try GPU batch?

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    # env = envs.ThorWrapperEnv(current_object_type='Microwave', interaction=False)
    # env = envs.ThorWrapperEnv(current_object_type='Microwave', dense_reward=True)
    # env = envs.ThorWrapperEnv(current_object_type='Mug')
    # env = envs.ThorWrapperEnv(current_object_type='Microwave', natural_language_instruction=True, grayscale=False)
    # shared_model = ActorCritic(
    #     env.observation_space.shape[0], env.action_space)
    # shared_model = ActorCritic(1, env.action_space)
    # shared_model = A3C_LSTM_GA(1, env.action_space).double()
    shared_model = A3C_LSTM_GA(3, 10).double()
    # shared_model = ActorCritic(3, env.action_space)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    # p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
    # p.start()
    # processes.append(p)
    rank = 0
    # train(rank, args, shared_model, counter, lock, optimizer)
    train_a3c_lstm_ga(rank, args, shared_model, counter, lock, optimizer)
