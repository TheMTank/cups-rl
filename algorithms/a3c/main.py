"""
Adapted from: https://github.com/ikostrikov/pytorch-a3c/blob/master/main.py
The main file needed within a3c. Runs of the train and test functions from their respective files.
Example of use:
`cd algorithms/a3c`
`python main.py`

Runs A3C on our AI2ThorEnv wrapper with default params (4 processes). Optionally it can be
run on any atari environment as well using the --atari and --atari-env-name params.
"""

from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp

from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from algorithms.a3c.envs import create_atari_env
from algorithms.a3c import my_optim
from algorithms.a3c.model import ActorCritic
from algorithms.a3c.test import test
from algorithms.a3c.train import train


# Based on: https://github.com/pytorch/examples/tree/master/mnist_hogwild
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
parser.add_argument('--test-sleep-time', type=int, default=200,
                    help='number of seconds to wait before testing again (default: 200)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')
parser.add_argument('-sync', '--synchronous', dest='synchronous', action='store_true',
                    help='Useful for debugging purposes e.g. import pdb; pdb.set_trace(). '
                         'Overwrites args.num_processes as everything is in main thread. '
                         '1 train() function is run and no test()')
parser.add_argument('-async', '--asynchronous', dest='synchronous', action='store_false')
parser.set_defaults(synchronous=False)

# Atari arguments. Good example of keeping code modular and allowing algorithms to run everywhere
parser.add_argument('--atari', dest='atari', action='store_true',
                    help='Run atari env instead with name below instead of ai2thor')
parser.add_argument('--atari-render', dest='atari_render', action='store_true',
                    help='Render atari')
parser.add_argument('--atari-env-name', default='PongDeterministic-v4',
                    help='environment to train on (default: PongDeterministic-v4)')
#
parser.set_defaults(atari=False)
parser.set_defaults(atari_render=False)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.atari:
        env = create_atari_env(args.atari_env_name)
        args.frame_dim = 42  # fixed to be 42x42 in envs.py _process_frame42()
    else:
        args.config_dict = {'max_episode_length': args.max_episode_length}
        env = AI2ThorEnv(config_dict=args.config_dict)
        args.frame_dim = env.config['resolution'][-1]
    shared_model = ActorCritic(env.observation_space.shape[0], env.action_space.n, args.frame_dim)
    shared_model.share_memory()

    env.close()  # above env initialisation was only to find certain params needed

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    if not args.synchronous:
        # test runs continuously and if episode ends, sleeps for args.test_sleep_time seconds
        p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
        p.start()
        processes.append(p)

        for rank in range(0, args.num_processes):
            p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        rank = 0
        # test(args.num_processes, args, shared_model, counter)  # for checking test functionality
        train(rank, args, shared_model, counter, lock, optimizer)  # run train on main thread
