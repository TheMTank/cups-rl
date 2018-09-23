"""
Adapted from VizDoom learning_pytorch.py
https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/learning_pytorch.py
"""

import time
import random
# from random import sample, randint, random

import skimage.color, skimage.transform
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import trange

import ai2thor.controller

# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
epochs = 20
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (64, 64)
# resolution = (300, 300)
# resolution = (30, 45)
episodes_to_watch = 10

model_savefile = "./model-doom.pth"
save_model = True
load_model = False
skip_learning = False

# Converts and down-samples the input image
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    gray = rgb2gray(img)
    return gray

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# DQN
class ReplayMemory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, channels, resolution[0], resolution[1])
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, 0, :, :] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, 0, :, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = random.sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]

class Net(nn.Module):
    def __init__(self, available_actions_count):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(648, 128)
        self.fc2 = nn.Linear(128, available_actions_count)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 648)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

criterion = nn.MSELoss()

def learn(s1, target_q):
    s1 = torch.from_numpy(s1)
    target_q = torch.from_numpy(target_q)
    s1, target_q = Variable(s1).double(), Variable(target_q).double()
    # import pdb;pdb.set_trace()
    output = model(s1)
    loss = criterion(output, target_q)
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def get_q_values(state):
    state = torch.from_numpy(state)
    state = Variable(state).double() #.unsqueeze(0).unsqueeze(1).double()
    assert len(list(state.size())) == 4
    return model(state)

def get_best_action(state):
    q = get_q_values(state)
    m, index = torch.max(q, 1)
    action = index.data.numpy()[0]
    return action


def learn_from_memory():
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)

        q = get_q_values(s2).data.numpy()
        q2 = np.max(q, axis=1)
        target_q = get_q_values(s1).data.numpy()
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        # import pdb; pdb.set_trace()
        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2
        learn(s1, target_q)


def perform_learning_step(epoch, event):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    s1 = preprocess(event.frame)

    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random.random() <= eps:
        a = random.randint(0, len(ACTION_SPACE) - 1)
    else:
        # Choose the best action according to the network.
        s1 = s1.reshape([1, 1, resolution[0], resolution[1]])
        a = get_best_action(s1)
    # reward = game.make_action(actions[a], frame_repeat)
    event, reward, isterminal = make_action(a, event)

    # isterminal = game.is_episode_finished()
    # isterminal = False
    # s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None
    s2 = preprocess(event.frame)

    # Remember the transition that was just experienced.
    # import pdb; pdb.set_trace()
    memory.add_transition(s1, a, s2, isterminal, reward)

    learn_from_memory()

    return event

# ai2thor setup

# action space stuff for ai2thor
ACTION_SPACE = {0: dict(action='MoveAhead'),
                1: dict(action='MoveBack'),
                2: dict(action='MoveRight'),
                3: dict(action='MoveLeft'),
                4: dict(action='LookUp'),
                5: dict(action='LookDown'),
                6: dict(action='RotateRight'),
                7: dict(action='RotateLeft'),
                # 1: dict(action='OpenObject'), # needs object id
                # 1: dict(action='CloseObject'), # needs object id
                8: dict(action='PickupObject'), # needs object id???
                9: dict(action='PutObject') # needs object id
                }

# also Teleport and TeleportFull but obviously only used for initialisation
NUM_ACTIONS = len(ACTION_SPACE.keys())

def calculate_reward(mug_id, task=0):
    # todo also just try endless reward and see if it spams picking up the cup.
    if task == 0:
        mugs_ids_collected_and_placed.add(mug_id)
        print('Reward collected!!!!!! {}'.format(mugs_ids_collected_and_placed))
        return 1 # has correctly picked up cup if we are here
    elif task == 1:
        if mug_id in mugs_ids_collected_and_placed:
            # already collected
            return 0
        else:
            mugs_ids_collected_and_placed.add(mug_id)
            print('Reward collected!!!!!! {}'.format(mugs_ids_collected_and_placed))
            return 1.0


def get_total_reward():
    return len(mugs_ids_collected_and_placed)

def make_action(action_int, event):
    reward = 0
    if action_int == 8:
        if len(event.metadata['inventoryObjects']) == 0:
            for o in event.metadata['objects']:
                if o['visible'] and (o['objectType'] == 'Mug'):
                    mug_id = o['objectId']
                    event = controller.step(
                        dict(action='PickupObject', objectId=mug_id), raise_for_failure=True)
                    reward = calculate_reward(mug_id)
                    break
    elif action_int == 9:
        # action = dict(action='PutObject', )
        if len(event.metadata['inventoryObjects']) > 0:

            for o in event.metadata['objects']:
                if o['visible'] and (o['objectType'] == 'CounterTop' or
                                     o['objectType'] == 'TableTop' or
                                     o['objectType'] == 'Sink' or
                                     o['objectType'] == 'CoffeeMachine' or
                                     o['objectType'] == 'Box'):
                    # import pdb;pdb.set_trace()
                    mug_id = event.metadata['inventoryObjects'][0]['objectId']
                    event = controller.step(dict(action='PutObject', objectId=mug_id, receptacleObjectId=o['objectId']),
                                            raise_for_failure=True)
                    reward = calculate_reward(mug_id)
                    break
    else:
        action = ACTION_SPACE[action_int]
        event = controller.step(action)

    return event, reward, is_episode_finished()

def is_episode_finished():
    return True if len(mugs_ids_collected_and_placed) == 3 else False

if __name__ == '__main__':
    controller = ai2thor.controller.Controller()
    controller.start()

    controller.reset('FloorPlan28')
    event = controller.step(dict(action='Initialize', gridSize=0.25))

    mugs = [obj for obj in event.metadata['objects'] if obj['objectType'] == 'Mug']
    mugs_ids_collected_and_placed = set()

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)

    if load_model:
        print("Loading model from: ", model_savefile)
        model = torch.load(model_savefile)
    else:
        model = Net(len(ACTION_SPACE)).double()

    optimizer = torch.optim.SGD(model.parameters(), learning_rate)

    print("Starting the training!")
    time_start = time.time()
    # random_a_space_int = random.randint(0, NUM_ACTIONS - 1)

    # event, reward = make_action(random_a_space_int, event)

    if not skip_learning:
        for epoch in range(epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []

            print("Training...")
            # game.new_episode()
            event = controller.step(dict(action='Initialize', gridSize=0.25))
            for learning_step in trange(learning_steps_per_epoch, leave=False):
                event = perform_learning_step(epoch, event)
                # if game.is_episode_finished():
                if is_episode_finished():
                    score = get_total_reward()
                    train_scores.append(score)
                    # game.new_episode()
                    event = controller.step(dict(action='Initialize', gridSize=0.25))
                    train_episodes_finished += 1

            # print("%d training episodes played." % train_episodes_finished)

            train_scores = np.array(train_scores)

            # print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
            #       "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

            print("\nTesting...")
            test_episode = []
            test_scores = []
            for test_episode in trange(test_episodes_per_epoch, leave=False):
                event = controller.step(dict(action='Initialize', gridSize=0.25))
                while not is_episode_finished():
                    state = preprocess(event.frame)
                    state = state.reshape([1, 1, resolution[0], resolution[1]])
                    best_action_index = get_best_action(state)
                    # todo fix bug here.
                    make_action(best_action_index, frame_repeat)
                r = get_total_reward()
                test_scores.append(r)

            # test_scores = np.array(test_scores)
            # print("Results: mean: %.1f +/- %.1f," % (
            #     test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
            #       "max: %.1f" % test_scores.max())

            # print("Saving the network weigths to:", model_savefile)
            # torch.save(model, model_savefile)

            print("Total elapsed time: %.2f minutes" % ((time.time() - time_start) / 60.0))


