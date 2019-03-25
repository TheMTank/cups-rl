"""
Different task implementations that can be defined inside an ai2thor environment
"""
from collections import Counter

from gym_ai2thor.utils import InvalidTaskParams


class TaskFactory:
    """
    Factory for tasks to be defined for a specific environment
    """
    @staticmethod
    def create_task(config):
        """
        Task factory method
        :param config: parsed config file
        :return: Task instance initialized
        """
        task_name = config['task']['task_name']
        if task_name == 'PickUp':
            # check that target objects are not selected as NON pickupables
            missing_objects = []
            for obj in config['task']['target_objects'].keys():
                if obj not in config['pickup_objects']:
                    missing_objects.append(obj)
            if missing_objects:
                raise InvalidTaskParams('Error initializing PickUpTask. The objects {} are not '
                                        'pickupable!'.format(missing_objects))
            else:
                return PickupTask(**config['task'])
        else:
            raise NotImplementedError('{} is not yet implemented!'.format(task_name))


class BaseTask:
    """
    Base class and factory for tasks to be defined for a specific environment
    """
    def __init__(self, config):
        self.task_config = config
        self.max_episode_length = config['max_episode_length'] \
            if 'max_episode_length' in config else 1000
        # default reward is negative to encourage the agent to move more
        self.movement_reward = config['movement_reward'] if 'movement_reward' in config else -0.01
        self.step_num = 0

        self.reset()

    def transition_reward(self, state):
        """
        Returns the reward given the corresponding information (state, dictionary with objects
        collected, distance to goal, etc.) depending on the task.
        :return: (args, kwargs) First elemnt represents the reward obtained at the step
                                Second element represents if episode finished at this step
        """
        raise NotImplementedError

    def reset(self):
        """

        :param args, kwargs: Configuration for task initialization
        :return:
        """
        raise NotImplementedError


class PickupTask(BaseTask):
    """
    This task consists on picking up an target object. Rewards are only collected if the right
    object was added to the inventory with the action PickUp (See gym_ai2thor.envs.ai2thor_env for
    details).
    """
    def __init__(self, target_objects=None, goal=None, **kwargs):
        self.object_rewards = target_objects if target_objects else {'Mug': 1}
        self.target_objects = self.object_rewards.keys()
        self.goal = Counter(goal if goal else {obj: float('inf') for obj in self.target_objects})
        self.pickedup_objects = Counter()
        self.prev_inventory = []
        super().__init__(kwargs)

    def transition_reward(self, state):
        reward, done = self.movement_reward, False
        curr_inventory = state.metadata['inventoryObjects']
        object_picked_up = not self.prev_inventory and curr_inventory and \
                           curr_inventory[0]['objectType'] in self.target_objects

        if object_picked_up:
            # One of the Target objects has been picked up
            self.pickedup_objects[curr_inventory[0]['objectType']] += 1
            # Add reward from the specific object
            reward += self.object_rewards[curr_inventory[0]['objectType']]
            print('{} reward collected!'.format(reward))

        if self.max_episode_length and self.step_num >= self.max_episode_length:
            print('Reached maximum episode length: {}'.format(self.step_num))
            done = True
        if self.goal == self.pickedup_objects:
            print('Reached goal at step {}'.format(self.step_num))
            done = True

        self.prev_inventory = state.metadata['inventoryObjects']
        return reward, done

    def reset(self):
        self.pickedup_objects = Counter()
        self.prev_inventory = []
        self.step_num = 0
