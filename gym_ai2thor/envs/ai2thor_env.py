"""
Base class implementation for ai2thor environments wrapper, which adds an openAI gym interface for
inheriting the predefined methods and can be extended for particular tasks.
"""
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
    def __init__(self,
                 scene_id='FloorPlan28',
                 seed=None,
                 grayscale=True,
                 config_path='gym_ai2thor/config_example.ini'):
        """
        :param scene_id:                (str)   Scene
        :param seed:                    (int)   Random seed
        :param grayscale:               (bool)  If True (default), transform RGB images to
                                                grayscale as part of the preprocessing
        :param config_path              (str)   Path to config file. Either absolute or relative to
                                                the root of this repository
        """
        # Loads config settings from file
        self.config = read_config(config_path)
        self.scene_id = scene_id
        self.controller = ai2thor.controller.Controller()
        self.controller.start()
        # Randomness settings
        self.np_random = None
        if seed:
            self.seed(seed)

        self.task = TaskFactory.create_task(self.config['task'])
        # Action settings
        if self.config['env']['interaction']:
            self.valid_actions = POSSIBLE_ACTIONS.copy()
        else:
            self.valid_actions = [action for action in POSSIBLE_ACTIONS
                                  if not action.endswith('Object')]
        self.action_names = tuple(action_str for action_str in self.valid_actions)
        self.action_space = spaces.Discrete(len(self.action_names))
        # Image settings
        self.event = None
        self.grayscale = grayscale
        self.resolution = (128, 128)  # (64, 64)
        if self.grayscale:
            self.observation_space = spaces.Box(low=0,
                                                high=255,
                                                shape=(self.resolution[0], self.resolution[1], 1),
                                                dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0,
                                                high=255,
                                                shape=(self.resolution[0], self.resolution[1], 3),
                                                dtype=np.uint8)

        # Object settings
        # acceptable objects taken from config.ini file. Stripping to allow spaces
        if self.config['env']['interaction']:
            self.objects =\
                {'pickupables': [x.strip() for x in self.config['env']['PICKUP_OBJECTS']
                    .split(',')],
                 'receptacles': [x.strip() for x in self.config['env']['ACCEPTABLE_RECEPTACLES']
                     .split(',')],
                 'openables':   [x.strip() for x in self.config['env']['OPENABLE_OBJECTS']
                     .split(',')]}
        self.reset()

    def step(self, action):
        if not isinstance(action, int):
            raise error.InvalidAction(f'Action must be an integer between '
                                      f'0 and {self.action_space.n}!')
        action = self.action_names[action]
        valid_action = False
        visible_objects = [obj for obj in self.event.metadata['objects'] if obj['visible']]
        if action.endswith('Object'):
            interaction_obj, distance = None, float('inf')
            if action == 'PutObject':
                closest_receptacle = None
                for obj in visible_objects:
                    if obj['receptacle'] and obj['distance'] < distance \
                        and obj in self.objects['receptacles'] \
                            and len(obj['receptacleObjectIds']) < obj['receptacleCount']:
                        closest_receptacle = obj
                        distance = closest_receptacle['distance']
                if self.event.metadata['inventoryObjects'] and closest_receptacle:
                    interaction_obj = closest_receptacle
                    self.event = self.controller.step(
                        dict(action=action,
                             objectId=self.event.metadata['inventoryObjects'][0],
                             receptacleObjectId=interaction_obj['objectId']))
            elif action == 'PickupObject':
                closest_pickupable = None
                for obj in visible_objects:
                    if obj['pickupable'] and obj['distance'] < distance and \
                            obj['name'] in self.objects['pickupables']:
                        closest_pickupable = obj
                if closest_pickupable and not self.event.metadata['inventoryObjects']:
                    interaction_obj = closest_pickupable
                    self.event = self.controller.step(
                        dict(action=action,
                             objectId=interaction_obj['objectId']))
            elif action == 'OpenObject':
                closest_openable = None
                for obj in visible_objects:
                    if obj['openable'] and obj['distance'] < distance and \
                            obj['name'] in self.objects['openables']:
                        closest_openable = obj
                        distance = closest_openable['distance']
                    if closest_openable:
                        interaction_obj = closest_openable
                        self.event = self.controller.step(
                            dict(action=action,
                                 objectId=interaction_obj['objectId']))
            elif action == 'CloseObject':
                closest_openable = None
                for obj in visible_objects:
                    if obj['openable'] and obj['distance'] < distance and obj['isopen'] and \
                            obj['name'] in self.objects['openables']:
                        closest_openable = obj
                        distance = closest_openable['distance']
                    if closest_openable:
                        interaction_obj = closest_openable
                        self.event = self.controller.step(
                            dict(action=action,
                                 objectId=interaction_obj['objectId']))
            else:
                raise error.InvalidAction(f'Invalid action {action}. '
                                          'You should never end up here anyways')
            if interaction_obj:
                valid_action = True
                print(f"{action}: {interaction_obj}. "
                      f"Inventory: {self.event.metadata['inventoryObjects']}")
        else:
            self.event = self.controller.step(dict(action=action))
            valid_action = True

        self.task.step_n += 1
        state = self.preprocess(self.event.frame)
        if valid_action:
            reward, done = self.task.calculate_reward(state)
        else:
            reward, done = None, False

        return state, reward, done

    def preprocess(self, img):
        # TODO: move this function to another script
        img = transform.resize(img, self.resolution)
        img = img.astype(np.float32)
        if self.grayscale:
            img = self.rgb2gray(img)  # todo cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def rgb2gray(self, rgb):
        # TODO: move this function to another script
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def reset(self):
        print('Resetting environment and starting new episode')
        self.controller.reset(self.scene_id)
        self.event = self.controller.step(dict(action='Initialize',
                                               gridSize=0.25,
                                               renderDepthImage=True,
                                               renderClassImage=True,
                                               renderObjectImage=True))
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
        pass


if __name__ == '__main__':
    AI2ThorEnv()
