import os
import json
import configparser
import sys

import numpy as np
import skimage.color, skimage.transform
import ai2thor.controller

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
    # Teleport and TeleportFull but these shouldn't be actions for an agent
]

class ThorWrapperEnv():
    def __init__(self, scene_id='FloorPlan28', seed=None, task=0, max_episode_length=1000, current_object_type='Mug',
                 grayscale=True, interaction=True, config_path='config_example.ini', movement_reward=0):
        # Loads config file from path relative to ai2thor_wrapper python package folder
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
        self.config = configparser.ConfigParser()
        config_output = self.config.read(config_path)
        if len(config_output) == 0:
            print('No config file found at: {}. Exiting'.format(config_path))
            sys.exit()

        self.scene_id = scene_id
        self.controller = ai2thor.controller.Controller()
        self.controller.start()
        if seed:
            self.seed(seed)

        self.controller.reset(self.scene_id)
        self.event = self.controller.step(dict(action='Initialize', gridSize=0.25, renderDepthImage=True,
                                               renderClassImage=True,
                                               renderObjectImage=True))

        self.current_object_type = current_object_type
        self.max_episode_length = max_episode_length
        self.t = 0
        self.done = False
        self.task = task
        self.movement_reward = movement_reward

        # action dictionary from int to dict action to pass to event.controller.step()
        self.ACTION_SPACE = {}
        for idx, action_str in enumerate(POSSIBLE_ACTIONS):
            self.ACTION_SPACE[idx] = dict(action=action_str)

        if not interaction:
            interaction_actions = ['OpenObject', 'CloseObject', 'PickupObject', 'PutObject']
            for idx in range(len(POSSIBLE_ACTIONS)):
                if self.ACTION_SPACE[idx]['action'] in interaction_actions:
                    self.ACTION_SPACE.pop(idx)

        self.NUM_ACTIONS = len(self.ACTION_SPACE.keys())
        self.action_space = self.NUM_ACTIONS

        # acceptable objects taken from config.ini file. Stripping to allow spaces
        self.pickup_object_types = [x.strip() for x in self.config['ENV_SPECIFIC']['PICKUP_OBJECTS'].split(',')]
        self.receptacle_object_types = [x.strip() for x in self.config['ENV_SPECIFIC']['ACCEPTABLE_RECEPTACLES'].split(',')]
        print('Objects that can be picked up: \n{}'.format(self.pickup_object_types))
        print('Objects that allow objects placed into/onto them (receptacles): \n{}'.format(self.receptacle_object_types))

        self.grayscale = grayscale
        self.resolution = (128, 128)  # (64, 64)
        if self.grayscale:
            self.observation_space = np.array((1,) + self.resolution)
        else:
            self.observation_space = np.array((3,) + self.resolution)

        self.goal_objects_collected_and_placed = []
        self.last_amount_of_goal_objects = len(self.goal_objects_collected_and_placed)

    def step(self, action_int):
        if self.ACTION_SPACE[action_int]['action'] == 'PickupObject':
            if len(self.event.metadata['inventoryObjects']) == 0:
                for o in self.event.metadata['objects']:
                    # loop through objects that are visible, pickupable and there is a bounding box visible
                    if o['visible'] and o['pickupable'] and o['objectType'] in self.pickup_object_types \
                            and o['objectId'] in self.event.instance_detections2D:
                        object_id = o['objectId']
                        object_type = o['objectType']
                        self.event = self.controller.step(
                            dict(action='PickupObject', objectId=object_id), raise_for_failure=True)
                        if object_type == self.current_object_type:
                            self.goal_objects_collected_and_placed.append(object_id)
                        print('Picked up', self.event.metadata['inventoryObjects'])
                        break
        elif self.ACTION_SPACE[action_int]['action'] == 'PutObject':
            if len(self.event.metadata['inventoryObjects']) > 0:
                for o in self.event.metadata['objects']:
                    # loop through receptacles
                    if o['visible'] and o['receptacle'] and \
                            o['objectType'] in self.receptacle_object_types and \
                            len(o['receptacleObjectIds']) < o['receptacleCount']:
                        # todo might still crash.
                        inventory_object_id = self.event.metadata['inventoryObjects'][0]['objectId']
                        inventory_object_type = self.event.metadata['inventoryObjects'][0]['objectType']

                        self.event = self.controller.step(dict(action='PutObject', objectId=inventory_object_id,
                                                               receptacleObjectId=o['objectId']))#,
                                                               #raise_for_failure=True)
                        if inventory_object_type == self.current_object_type:
                            self.goal_objects_collected_and_placed.remove(inventory_object_id)
                        print('Placed', inventory_object_id, ' onto', o['objectId'], ' Inventory: ',
                              self.event.metadata['inventoryObjects'])
                        break
        elif self.ACTION_SPACE[action_int]['action'] == 'OpenObject':
            for o in self.event.metadata['objects']:
                # loop through objects that are visible, openable, closed
                if o['visible'] and o['openable'] and not o['isopen'] and o['objectType'] == 'Microwave' \
                        and o['objectId'] in self.event.instance_detections2D:
                    print('Opened', o['objectId'])

                    self.event = self.controller.step(
                        dict(action='OpenObject', objectId=o['objectId']), raise_for_failure=True)
                    break
        elif self.ACTION_SPACE[action_int]['action'] == 'CloseObject':
            for o in self.event.metadata['objects']:
                # loop through objects that are visible, openable, open
                if o['visible'] and o['openable'] and o['isopen'] and o['objectType'] == 'Microwave' \
                        and o['objectId'] in self.event.instance_detections2D:
                    print('Closed', o['objectId'])
                    self.event = self.controller.step(
                        dict(action='CloseObject', objectId=o['objectId']), raise_for_failure=True)
                    break
        else:
            action = self.ACTION_SPACE[action_int]
            self.event = self.controller.step(action)

        self.t += 1
        state = self.preprocess(self.event.frame)
        reward = self.calculate_reward(self.done)
        self.done = self.is_episode_finished()
        return state, reward, self.done

    def preprocess(self, img):
        img = skimage.transform.resize(img, self.resolution)
        img = img.astype(np.float32)
        if self.grayscale:
            img = self.rgb2gray(img)  # todo cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def calculate_reward(self, done=False):
        reward = self.movement_reward
        if self.task == 0:
            if self.last_amount_of_goal_objects < len(self.goal_objects_collected_and_placed):
                self.last_amount_of_goal_objects = len(self.goal_objects_collected_and_placed)
                # mug has been picked up
                reward += 1
                print('{} reward collected! Inventory: {}'.format(reward, self.goal_objects_collected_and_placed))
            elif self.last_amount_of_goal_objects > len(self.goal_objects_collected_and_placed):
                # placed mug onto/into receptacle
                pass
            self.last_amount_of_goal_objects = len(self.goal_objects_collected_and_placed)
        else:
            raise NotImplementedError

        return reward

    def is_episode_finished(self):
        if self.max_episode_length and self.t >= self.max_episode_length:
            print('Reached maximum episode length: t: {}'.format(self.t))
            return True

    def reset(self):
        print('Resetting environment and starting new episode')
        self.t = 0
        self.controller.reset(self.scene_id)
        self.event = self.controller.step(dict(action='Initialize', gridSize=0.25, renderDepthImage=True,
                                               renderClassImage=True,
                                               renderObjectImage=True))
        self.goal_objects_collected_and_placed = []
        self.last_amount_of_goal_objects = len(self.goal_objects_collected_and_placed)
        self.done = False
        state = self.preprocess(self.event.frame)
        return state

    def seed(self, seed):
        raise NotImplementedError
        # self.random_seed = seed  # no effect unless RandomInitialize is called
