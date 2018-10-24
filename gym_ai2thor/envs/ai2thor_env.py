"""
Base class implementation for ai2thor environments wrapper, which adds an openAI gym interface for
inheriting the predefined methods and can be extended for particular tasks.
"""
import os
import configparser

import ai2thor.controller
import numpy as np
from skimage import transform

import gym
from gym import error, spaces
from gym.utils import seeding
from gym_ai2thor.envs.utils import read_config
from gym_ai2thor.tasks import TaskFactory

POSSIBLE_ACTIONS = [
    'MoveAhead',
    'MoveBack',
    'MoveRight',
    'MoveLeft',
    'LookUp',
    'LookDown',
    'RotateRight',
    'RotateLeft',
    'OpenObject',
    'CloseObject',
    'PickupObject',
    'PutObject'
    # Teleport and TeleportFull but these shouldn't be allowable actions for an agent
]


class AI2ThorEnv(gym.Env):
    """
    Wrapper base class
    """
    def __init__(self, scene_id='FloorPlan28',
                 seed=None,
                 grayscale=True,
                 config_path='gym_ai2thor/config_example.ini'):
        """
        :param scene_id:                ()    Scene
        :param seed:                    ()    Random seed
        :param task:                    (str) Task descriptor
        :param grayscale:
        """
        # Loads config file from path relative to gym_ai2thor python package folder
        self.config = read_config(config_path)
        self.scene_id = scene_id
        self.controller = ai2thor.controller.Controller()
        self.controller.start()

        self.np_random = None
        if seed:
            self.seed(seed)

        self.step_n = 0
        self.done = False
        self.task = TaskFactory(self.config['task'])

        # action dictionary from int to dict action to pass to event.controller.step()
        self.valid_actions = POSSIBLE_ACTIONS if self.config['interaction'] else POSSIBLE_ACTIONS[:-4]
        self.action_names = tuple(action_str for action_str in self.valid_actions)
        self.action_space = spaces.Tuple((spaces.Discrete(len(self.action_names))))

        # acceptable objects taken from config.ini file. Stripping to allow spaces
        self.pickup_object_types = \
            [x.strip() for x in self.config['ENV_SPECIFIC']['PICKUP_OBJECTS'].split(',')]
        self.receptacle_object_types = \
            [x.strip() for x in self.config['ENV_SPECIFIC']['ACCEPTABLE_RECEPTACLES'].split(',')]
        self.openable_objects = \
            [x.strip() for x in self.config['ENV_SPECIFIC']['OPENABLE_OBJECTS'].split(',')]
        if self.interaction:
            print('Objects that can be picked up: \n{}'.format(self.pickup_object_types))
            print('Objects that allow objects placed into/onto them (receptacles): \n{}'.
                  format(self.receptacle_object_types))
            print('Objects that can be opened: \n{}'.format(self.openable_objects))

        self.grayscale = grayscale
        self.resolution = (128, 128)  # (64, 64)
        if self.grayscale:
            self.observation_space = np.array((1,) + self.resolution)
        else:
            self.observation_space = np.array((3,) + self.resolution)

        self.goal_objects_collected_and_placed = []
        self.last_amount_of_goal_objects = len(self.goal_objects_collected_and_placed)

        self.reset()

    def step(self, a):
        if self.action_tuple[a] == 'PickupObject':
            if not self.event.metadata['inventoryObjects']:
                for obj in self.event.metadata['objects']:
                    # loop through objects that are visible, pickupable and there is a bounding
                    # box visible
                    if obj['visible'] and \
                           obj['pickupable'] and \
                           obj['objectType'] in self.pickup_object_types and \
                           obj['objectId'] in self.event.instance_detections2D:
                        object_id = obj['objectId']
                        object_type = obj['objectType']
                        self.event = self.controller.step(
                            dict(action='PickupObject', objectId=object_id), raise_for_failure=True)
                        if object_type == self.current_task_object:
                            self.goal_objects_collected_and_placed.append(object_id)
                        print('Picked up', self.event.metadata['inventoryObjects'])
                        break
        elif self.action_tuple[a] == 'PutObject':
            if self.event.metadata['inventoryObjects']:
                for obj in self.event.metadata['objects']:
                    # loop through receptacles
                    if obj['visible'] and \
                            obj['receptacle'] and \
                            obj['objectType'] in self.receptacle_object_types and \
                            len(obj['receptacleObjectIds']) < obj['receptacleCount']:
                        # todo might still crash.
                        inventory_object_id = \
                            self.event.metadata['inventoryObjects'][0]['objectId']
                        inventory_object_type = \
                            self.event.metadata['inventoryObjects'][0]['objectType']

                        self.event = self.controller.step(
                            dict(action='PutObject', objectId=inventory_object_id,
                                 receptacleObjectId=obj['objectId']))  #, raise_for_failure=True)
                        # if inventory_object_type == self.current_task_object:

                        if self.goal_objects_collected_and_placed:
                            pass
                            # import pdb;pdb.set_trace() # todo why?
                        # self.goal_objects_collected_and_placed.remove(inventory_object_id)
                        self.goal_objects_collected_and_placed = []
                        print('Placed', inventory_object_id, ' onto', obj['objectId'],
                              ' Inventory: ', self.event.metadata['inventoryObjects'])
        elif self.action_tuple[a] == 'OpenObject':
            for obj in self.event.metadata['objects']:
                # loop through objects that are visible, openable, closed
                if obj['visible'] and obj['openable'] and \
                        not obj['isopen'] and obj['objectType'] in self.openable_objects and \
                        obj['objectId'] in self.event.instance_detections2D:
                    print('Opened', obj['objectId'])

                    self.event = self.controller.step(
                        dict(action='OpenObject',
                             objectId=obj['objectId']),
                        raise_for_failure=True)
        elif self.action_tuple[a] == 'CloseObject':
            for obj in self.event.metadata['objects']:
                # loop through objects that are visible, openable, open
                if obj['visible'] and obj['openable'] and obj['isopen'] and \
                        obj['objectType'] in self.openable_objects and \
                        obj['objectId'] in self.event.instance_detections2D:
                    print('Closed', obj['objectId'])
                    self.event = self.controller.step(
                        dict(action='CloseObject', objectId=obj['objectId']),
                        raise_for_failure=True)
        else:
            action = self.action_tuple[a]
            self.event = self.controller.step(dict(action=action))

        self.step_n += 1
        state = self.preprocess(self.event.frame)
        reward = self.calculate_reward(self.done)
        self.done = self.episode_finished()
        return state, reward, self.done

    def preprocess(self, img):
        img = transform.resize(img, self.resolution)
        img = img.astype(np.float32)
        if self.grayscale:
            img = self.rgb2gray(img)  # todo cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def reset(self):
        print('Resetting environment and starting new episode')
        self.t = 0
        self.controller.reset(self.scene_id)
        self.event = self.controller.step(dict(action='Initialize',
                                               gridSize=0.25,
                                               renderDepthImage=True,
                                               renderClassImage=True,
                                               renderObjectImage=True))
        self.goal_objects_collected_and_placed = []
        self.last_amount_of_goal_objects = len(self.goal_objects_collected_and_placed)
        self.done = False
        state = self.preprocess(self.event.frame)
        return state

    def render(self, mode='human'):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        return seed1

    def close(self):
        pass


if __name__ == '__main__':
    AI2ThorEnv()
