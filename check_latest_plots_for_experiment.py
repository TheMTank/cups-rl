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
    idx_of_latest = path_filename_ints.index(max(path_filename_ints))
    path_to_return = list_of_paths[idx_of_latest]
    print('Found latest plot to show: {}'.format(path_to_return))
    return path_to_return

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--experiment-id', type=str, default='6a163e74-4c96-494d-9490-0157bb93e154',
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

    fp1 = get_path_of_latest_plot(avgavg_paths)
    fp2 = get_path_of_latest_plot(total_reward_paths)
    fp3 = get_path_of_latest_plot(episode_length_paths)

    figsize = (12, 12)
    img = mpimg.imread(fp3)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()

    img = mpimg.imread(fp2)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()

    img = mpimg.imread(fp1)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()
