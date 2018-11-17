import time
import os
import glob
import argparse


# import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
# mpl.use('Agg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# from envs import create_atari_env
import envs
from model import ActorCritic, ActorCriticExtraInput
import utils

def get_path_of_latest_plot(list_of_paths):
    # path_filename_ints = [int(x.split('/')[-1].split('.pth.tar')[0].split('_')[-1]) for x in list_of_paths]
    path_filename_ints = [int(x.split('-')[-1].split('.png')[0]) for x in list_of_paths]
    if path_filename_ints:
        idx_of_latest = path_filename_ints.index(max(path_filename_ints))
        path_to_return = list_of_paths[idx_of_latest]
        print('Found latest plot to show: {}'.format(path_to_return))
    else:
        print('Did not find path with input list as: {}'.format(list_of_paths))
        path_to_return = False
    return path_to_return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show latest plots for experiment id')
    parser.add_argument('--experiment-id',
                        # type=str, default='testing',
                        # type=str, default='microwave_or_cup_two_one_word_sentence_1.5million_steps',
                        # type=str, default='FloorPlan27-over-1mil-steps-almost-optimal', # first time new room
                        type=str, default='First-Time-FloorPlan26-8-processes-all-night',
                        # type=str, default='77d9f492-8f1e-4dff-b588-2d4cbee85591',
                        help='experiment-id')
    args = parser.parse_args()

    experiment_path = '/home/beduffy/all_projects/ai2thor-testing/experiments/{}'.format(args.experiment_id)
    if not os.path.exists(experiment_path):
        print('Could not find experiment folder: {}'.format(experiment_path))
    else:
        print('Experiment already exists at path: {}'.format(experiment_path))

        avgavg_paths = glob.glob(experiment_path + '/a3c-avgavg-reward-*')
        total_reward_paths = glob.glob(experiment_path + '/a3c-total-*')
        episode_length_paths = glob.glob(experiment_path + '/episode-lengths-*')
        value_losses = glob.glob(experiment_path + '/value-loss-*')
        policy_losses = glob.glob(experiment_path + '/policy-loss-*')

        fp1 = get_path_of_latest_plot(avgavg_paths)
        fp2 = get_path_of_latest_plot(total_reward_paths)
        fp3 = get_path_of_latest_plot(episode_length_paths)
        fp4 = get_path_of_latest_plot(value_losses)
        fp5 = get_path_of_latest_plot(policy_losses)

        # todo could put everything into a list or... could go visdom or tensorboard route

        if fp3:
            figsize = (12, 12)
            img = mpimg.imread(fp3)
            plt.figure(figsize=figsize)
            plt.imshow(img)
            plt.show()
        else:
            print('No fp3')

        if fp2:
            img = mpimg.imread(fp2)
            plt.figure(figsize=figsize)
            plt.imshow(img)
            plt.show()
        else:
            print('No fp2')

        if fp1:
            img = mpimg.imread(fp1)
            plt.figure(figsize=figsize)
            plt.imshow(img)
            plt.show()
        else:
            print('No fp1')

        if fp4:
            img = mpimg.imread(fp4)
            plt.figure(figsize=figsize)
            plt.imshow(img)
            plt.show()
        else:
            print('No fp4')

        if fp5:
            img = mpimg.imread(fp5)
            plt.figure(figsize=figsize)
            plt.imshow(img)
            plt.show()
        else:
            print('No fp5')
