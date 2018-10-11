from __future__ import print_function

import argparse
import os
import uuid
import sys
import glob

import torch
import torch.multiprocessing as mp

import my_optim
import envs
from model import ActorCritic, ActorCriticExtraInput
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
parser.add_argument('--total-length', type=int, default=0, help='initial length if resuming')
parser.add_argument('--experiment-id', default=uuid.uuid4(),
                    help='random guid for separating plots and checkpointing. If experiment taken, '
                         'will resume training!')
parser.add_argument('--num-processes', type=int, default=1,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = "" # todo change, try GPU batch?

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    # shared_model = ActorCritic(env.observation_space.shape[0], env.action_space)
    # shared_model = ActorCritic(1, env.action_space)
    # shared_model = A3C_LSTM_GA(1, env.action_space).double()
    # shared_model = A3C_LSTM_GA(3, 10).double()
    # shared_model = A3C_LSTM_GA(3, 2).double()
    shared_model = ActorCritic(1, 10) # todo get 1 and 10 from environment without instantiating it
    # shared_model = ActorCriticExtraInput(1, 10)

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    experiment_path = '/home/beduffy/all_projects/ai2thor-testing/experiments/{}'.format(args.experiment_id)
    if not os.path.exists(experiment_path):
        print('Creating experiment folder: {}'.format(experiment_path))
        os.makedirs(experiment_path)
    else:
        print('Experiment already exists at path: {}'.format(experiment_path))
        checkpoint_paths = glob.glob(experiment_path + '/checkpoint*')
        # Take checkpoint path with most experience
        checkpoint_file_name_ints = [int(x.split('/')[-1].split('.pth.tar')[0].split('_')[-1])
                                     for x in checkpoint_paths]
        idx_of_latest = checkpoint_file_name_ints.index(max(checkpoint_file_name_ints))
        checkpoint_to_load = checkpoint_paths[idx_of_latest]
        print('Loading latest checkpoint: {}'.format(checkpoint_to_load))

        if os.path.isfile(checkpoint_to_load):
            print("=> loading checkpoint '{}'".format(checkpoint_to_load))
            checkpoint = torch.load(checkpoint_to_load)
            args.total_length = checkpoint['total_length']
            shared_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (total_length {})"
                  .format(checkpoint_to_load, checkpoint['total_length']))

        # todo have choice of checkpoint as well? args.resume could override the above

    shared_model.share_memory()
    processes = []
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    # p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
    # p.start()
    # processes.append(p)
    rank = 0
    # train(rank, args, shared_model, counter, lock, optimizer)
    train_a3c_lstm_ga(rank, args, shared_model, counter, lock, optimizer)
