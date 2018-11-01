"""
Different task implementations that can be defined inside an ai2thor environment
"""
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
            if config['task']['target_object'] in config['env']['pickup_objects']:
                return PickupTask(**config['task'])
            else:
                raise InvalidTaskParams('Error initializing PickUpTask. {} is not ' \
                          'pickupable!'.format(config['task']['target_object']))
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
        self.movement_reward = config['movement_reward'] if 'movement_reward' in config else 0
        self.step_num = 0

        self.reset()

    def transition_reward(self, *args, **kwargs):
        """
        Returns the reward given the corresponding information (state, dictionary with objects
        collected, distance to goal, etc.) depending on the task.
        :return: (args, kwargs) First elemnt represents the reward obtained at the step
                                Second element represents if episode finished at this step
        """
        raise NotImplementedError

    def reset(self, *args, **kwargs):
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
    def __init__(self, target_objects=('Mug',), terminations=(1,), **kwargs):
        self.target_objects = target_objects
        self.terminations = terminations
        self.moved_objects = {obj: 0 for obj in self.target_objects}
        self.object_rewards = {obj: 1 for obj in self.target_objects}
        super().__init__(kwargs)

    def transition_reward(self, prev_state, post_state):
        done = False
        interacted_obj = None
        reward = self.movement_reward
        picked_target = not prev_state.metadata['inventoryObjects'] and \
                        post_state.metadata['inventoryObjects'] and \
                        post_state.metadata['inventoryObjects'][0]['objectType'] \
                        in self.target_objects
        put_target = prev_state.metadata['inventoryObjects'] and \
                     not post_state.metadata['inventoryObjects'] and \
                     prev_state.metadata['inventoryObjects']['objectType'] in self.target_objects

        if picked_target:
            interacted_obj = post_state.metadata['inventoryObjects'][0]['objectType']
        elif put_target:
            interacted_obj = prev_state.metadata['inventoryObjects'][0]['objectType']
            # Target object has been picked up or put on a receptacle
        if interacted_obj:
            self.moved_objects[interacted_obj] += 1
            reward += self.object_rewards[interacted_obj]
            print('{}: {}. {} reward collected!'.format(post_state.metadata['lastAction'],
                                                        interacted_obj, reward))

        if self.max_episode_length and self.step_num >= self.max_episode_length:
            print('Reached maximum episode length: {}'.format(self.step_num))
            done = True

        return reward, done

    def reset(self):
        self.moved_objects = {obj: 0 for obj in self.target_objects}
        self.step_num = 0
