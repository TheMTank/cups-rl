"""
Base class implementation for ai2thor environments wrapper, which adds an openAI gym interface for
inheriting the predefined methods and can be extended for particular tasks.
"""
import ai2thor.controller
import numpy as np
from skimage import transform
from copy import deepcopy

import gym
from gym import error, spaces
from gym.utils import seeding
from gym_ai2thor.image_processing import rgb2gray
from gym_ai2thor.utils import read_config
from gym_ai2thor.tasks import TaskFactory

ALL_POSSIBLE_ACTIONS = [
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
    def __init__(self, seed=None, config_file='config_files/config_example.json', config_dict=None):
        """
        :param seed:         (int)   Random seed
        :param config_file:  (str)   Path to environment configuration file. Either absolute or
                                     relative path to the root of this repository.
        :param: config_dict: (dict)  Overrides specific fields from the input configuration file.
        """
        # Loads config settings from file
        self.config = read_config(config_file, config_dict)
        self.scene_id = self.config['scene_id']
        # Randomness settings
        self.np_random = None
        if seed:
            self.seed(seed)
        # Object settings
        # acceptable objects taken from config file.
        if self.config['pickup_put_interaction'] or \
                            self.config['open_close_interaction']:
            self.objects = {'pickupables': self.config['pickup_objects'],
                            'receptacles': self.config['acceptable_receptacles'],
                            'openables':   self.config['openable_objects']}
        # Action settings
        self.action_names = tuple(ALL_POSSIBLE_ACTIONS.copy())
        # remove open/close and pickup/put actions if respective interaction bool is set to False
        if not self.config['open_close_interaction']:
            # Don't allow opening and closing if set to False
            self.action_names = tuple([action_name for action_name in self.action_names if 'Open'
                                       not in action_name and 'Close' not in action_name])
        if not self.config['pickup_put_interaction']:
            self.action_names = tuple([action_name for action_name in self.action_names if 'Pickup'
                                       not in action_name and 'Put' not in action_name])
        self.action_space = spaces.Discrete(len(self.action_names))
        # Image settings
        self.event = None
        channels = 1 if self.config['grayscale'] else 3
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(channels, self.config['resolution'][0],
                                                   self.config['resolution'][1]),
                                            dtype=np.uint8)
        # ai2thor initialise settings
        self.scene_id = 'FloorPlan1'  # todo overriding above!
        self.cameraY = self.config.get('cameraY', -0.85)  # todo
        self.gridSize = self.config.get('gridSize', 0.01)  # todo?
        # rotation settings
        self.incremental_rotation_mode = self.config.get('incremental_rotation', True)  # todo change
        self.absolute_rotation = 0.0
        self.rotation_amount = 10.0
        # Create task from config
        self.task = TaskFactory.create_task(self.config)
        # Start ai2thor
        self.controller = ai2thor.controller.Controller()
        self.controller.local_executable_path = self.config.get('build_path',
            '/home/beduffy/all_projects/ai2thor/unity/build-test.x86_64')
        self.controller.start()

    def step(self, action, verbose=True):
        if not self.action_space.contains(action):
            raise error.InvalidAction('Action must be an integer between '
                                      '0 and {}!'.format(self.action_space.n))
        action_str = self.action_names[action]
        visible_objects = [obj for obj in self.event.metadata['objects'] if obj['visible']]

        if action_str.endswith('Object'):  # All interactions end with 'Object'
            # Interaction actions
            interaction_obj, distance = None, float('inf')
            inventory_before = self.event.metadata['inventoryObjects'][0]['objectType'] \
                if self.event.metadata['inventoryObjects'] else []
            if action_str == 'PutObject':
                closest_receptacle = None
                for obj in visible_objects:
                    # look for closest receptacle to put object from inventory
                    if obj['receptacle'] and obj['distance'] < distance \
                        and obj['objectType'] in self.objects['receptacles'] \
                            and len(obj['receptacleObjectIds']) < obj['receptacleCount']:
                        closest_receptacle = obj
                        distance = closest_receptacle['distance']
                if self.event.metadata['inventoryObjects'] and closest_receptacle:
                    interaction_obj = closest_receptacle
                    self.event = self.controller.step(
                            dict(action=action_str,
                                 objectId=self.event.metadata['inventoryObjects'][0]['objectId'],
                                 receptacleObjectId=interaction_obj['objectId']))
            elif action_str == 'PickupObject':
                closest_pickupable = None
                for obj in visible_objects:
                    # look for closest object to pick up
                    if obj['pickupable'] and obj['distance'] < distance and \
                            obj['objectType'] in self.objects['pickupables']:
                        closest_pickupable = obj
                if closest_pickupable and not self.event.metadata['inventoryObjects']:
                    interaction_obj = closest_pickupable
                    self.event = self.controller.step(
                        dict(action=action_str, objectId=interaction_obj['objectId']))
            elif action_str == 'OpenObject':
                closest_openable = None
                for obj in visible_objects:
                    # look for closest closed receptacle to open it
                    if obj['openable'] and obj['distance'] < distance and not obj['isopen'] and \
                            obj['objectType'] in self.objects['openables']:
                        closest_openable = obj
                        distance = closest_openable['distance']
                    if closest_openable:
                        interaction_obj = closest_openable
                        self.event = self.controller.step(
                            dict(action=action_str,
                                 objectId=interaction_obj['objectId']))
            elif action_str == 'CloseObject':
                closest_openable = None
                for obj in visible_objects:
                    # look for closest opened receptacle to close it
                    if obj['openable'] and obj['distance'] < distance and obj['isopen'] and \
                            obj['objectType'] in self.objects['openables']:
                        closest_openable = obj
                        distance = closest_openable['distance']
                    if closest_openable:
                        interaction_obj = closest_openable
                        self.event = self.controller.step(
                            dict(action=action_str,
                                 objectId=interaction_obj['objectId']))
            else:
                raise error.InvalidAction('Invalid interaction {}'.format(action_str))
            if interaction_obj and verbose:
                inventory_after = self.event.metadata['inventoryObjects'][0]['objectType'] \
                    if self.event.metadata['inventoryObjects'] else []
                if action_str in ['PutObject', 'PickupObject']:
                    inventory_changed_str = 'Inventory before/after: {}/{}.'.format(
                                                            inventory_before, inventory_after)
                else:
                    inventory_changed_str = ''
                print('{}: {}. {}'.format(
                    action_str, interaction_obj['objectType'], inventory_changed_str))
        elif 'Rotate' in action_str:
            import pdb;pdb.set_trace()
            if self.incremental_rotation_mode:
                # Rotate actions
                if 'Left' in action_str:
                    self.absolute_rotation -= self.rotation_amount
                    self.event = self.controller.step(
                        dict(action='Rotate', rotation=self.absolute_rotation))
                elif 'Right' in action_str:
                    self.absolute_rotation += self.rotation_amount
                    self.event = self.controller.step(
                        dict(action='Rotate', rotation=self.absolute_rotation))
            else:
                # Do normal RotateLeft command
                self.event = self.controller.step(dict(action=action_str))
        else:
            # Move and Look actions
            self.event = self.controller.step(dict(action=action_str))

        self.task.step_num += 1
        state_image = self.preprocess(self.event.frame)
        reward, done = self.task.transition_reward(self.event)
        info = {}

        return state_image, reward, done, info

    def preprocess(self, img):
        """
        Compute image operations to generate state representation
        """
        img = transform.resize(img, self.config['resolution'], mode='reflect')
        img = img.astype(np.float32)
        if self.observation_space.shape[0] == 1:
            img = rgb2gray(img)  # todo cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def reset(self):
        print('Resetting environment and starting new episode')
        self.controller.reset(self.scene_id)
        self.event = self.controller.step(dict(action='Initialize', gridSize=self.gridSize,
                                               cameraY=self.cameraY, renderDepthImage=True,
                                               renderClassImage=True, renderObjectImage=True))
        self.task.reset()
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
        self.controller.stop()


if __name__ == '__main__':
    AI2ThorEnv()
