"""
Adapted from https://github.com/ikostrikov/pytorch-a3c
"""

from __future__ import print_function

import time
import argparse
import os
import uuid
import glob

import torch
import torch.multiprocessing as mp

import my_optim
import envs
from model import ActorCritic
from a3c_lstm_ga_model import A3C_LSTM_GA
from test import test
from train import train
from train_a3c_lstm_ga import train_a3c_lstm_ga

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0003,
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
parser.add_argument('--number-of-episodes', type=int, default=0, help='number-of-episodes passed if resuming')
parser.add_argument('-eid', '--experiment-id', default=uuid.uuid4(),
                    help='random or chosen guid for folder creation for plots and checkpointing. If experiment taken, '
                         'will resume training!')
parser.add_argument('--num-processes', type=int, default=2,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000, # todo doesn't affect below. fix.
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'  # todo try multiple threads
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # "" # todo change, try GPU batch? or off-policy algorithm

    args = parser.parse_args()

    # todo print args to file in experiment folder
    # todo print all logs to experiment folder
    # todo allow changing name of experiment folder and perfect checkpointing
    # todo load episode number, learning rate and optimiser and more

    torch.manual_seed(args.seed)
    # shared_model = ActorCritic(env.observation_space.shape[0], env.action_space)
    # shared_model = ActorCritic(1, env.action_space)
    # shared_model = A3C_LSTM_GA(1, env.action_space).double()
    shared_model = A3C_LSTM_GA(3, 8).double()  # todo try GPU
    # shared_model = A3C_LSTM_GA(3, 2).double()
    # shared_model = ActorCritic(1, 10) # todo get 1 and 10 from environment without instantiating it
    # shared_model = ActorCriticExtraInput(1, 10)

    # todo train 8 processes but on a new kitchen each time (with it possibly changing everytime or
    # maybe one process stays on its own kitchen)

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
        if checkpoint_file_name_ints:
            idx_of_latest = checkpoint_file_name_ints.index(max(checkpoint_file_name_ints))
            checkpoint_to_load = checkpoint_paths[idx_of_latest]
            print('Loading latest checkpoint: {}'.format(checkpoint_to_load))

            if os.path.isfile(checkpoint_to_load):
                print("=> loading checkpoint '{}'".format(checkpoint_to_load))
                checkpoint = torch.load(checkpoint_to_load)
                args.total_length = checkpoint['total_length']
                shared_model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])  # todo check if overwrites learning rate. probably does

                if checkpoint['number_of_episodes']:
                    args.number_of_episodes = checkpoint['number_of_episodes']

                if checkpoint['counter']:
                    checkpoint_counter = checkpoint['counter']

                for param_group in optimizer.param_groups:
                    print('Learning rate: ', param_group['lr'])  # oh it doesn't

                print("=> loaded checkpoint '{}' (total_length {})"
                      .format(checkpoint_to_load, checkpoint['total_length']))
        else:
            raise ValueError('No checkpoints saved in this directory')

        # todo have choice of checkpoint as well? args.resume could override the above

    shared_model.share_memory()
    processes = []
    counter = mp.Value('i', checkpoint_counter if checkpoint_counter else 0)
    lock = mp.Lock()

    # p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
    # p.start()
    # processes.append(p)

    start = time.time()
    for rank in range(0, args.num_processes):
        # p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        p = mp.Process(target=train_a3c_lstm_ga, args=(rank, args, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print('Total time taken (for all processes): {}'.format(time.time() - start))
    # todo close aithor apps here properly to save memory and cpu

    # Time taken for total length over 1000 (altogether) for:
    # 1 process = 98.99s (>1000) (1.8avg seconds per 20. Estimated: 50 * 1.8 = 90 seconds)
    # 2 process = 62.24s (>500) (2.18avg seconds per 20. Estimated: 25 * 2.18 = 54.5 seconds)
    # 4 process = 46.86s (>250) (3avg seconds per 20. Estimated: 12.5 * 3 = 37.5 seconds)
    # 8 process = 47.52s (>125) (5.3avg seconds per 20. Estimated: 6.25 * 5.3 = 33.125 seconds)
    # 16 process = 69.17s (>63) (12avg seconds per 20. Estimated: 3.125 * 12 = 37.5 seconds)

    # Time taken for total length over 10000 (altogether) for:
    # 1 process = 966.99s (1.8avg seconds per 20. Estimated: 500 * 1.8 = 900 seconds)
    # 8 process = 373.27s (5.7avg seconds per 20. Estimated: 62.5 * 5.7 = 356.25 seconds)
    # 12 process = 390.20 (>833) (8.2avg seconds per 20. Estimated: 41.65 * 8.2 = 341.53 seconds)
    # 16 process = 404.93s (>625) (12avg seconds per 20. Estimated: 31.25 * 12 = 375 seconds)

    # Using above times to then estimate time for 1 million steps:
    # 1 process = (1.8avg seconds per 20. Estimated: 50000 * 1.8 = 90000 seconds/25 hours)
    # 8 process = (avg seconds per 20. Estimated: 6250 * 5.7 = 35625 seconds/9.89 hours)
    # 16 process = (12avg seconds per 20. Estimated: 3125 * 12 = 37500 seconds/10.41 hours)
