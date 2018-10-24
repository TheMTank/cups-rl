"""
Different task implementations that can be defined inside an ai2thor environment
"""


class TaskFactory:
    """
    Base class and factory for tasks to be defined for a specific environment
    """
    def __init__(self, config):
        self.max_episode_length = 1000
        self.current_task_object = 'Mug'
        self.interaction = True
        self.env_config = config
        self.movement_reward = 0
        self.task_name = config['task_name']

    @staticmethod
    def create_task(config):
        """
        Task factory method
        :param config: parsed config file
        :return: Task instance initialized
        """
        task_name = config['task_name']
        if task_name == 'PickUp':
            return PickupTask(task_name)
        else:
            raise NotImplementedError(f'{task_name} is not yet implemented!')

    def calculate_reward(self, info):
        """
        Returns the reward given the corresponding information (state, dictionary with objects
        collected, distance to goal, etc.) depending on the task.
        :return: (int, bool) First elemnt represents the reward obtained at the step
                             Second element represents if episode finished at this step
        """
        raise NotImplementedError


class PickupTask(TaskFactory):
    """
    This task consists on picking up an target object. Rewards are only collected if the right
    object was added to the inventory with the action PickUp (See gym_ai2thor.envs.ai2thor_env for
    details).
    """
    def __init__(self, target_object, termination='first'):
        self.target_object = target_object
        self.termination = termination
        super().__init__()

    def calculate_reward(self):
        done = False
        reward = self.movement_reward
        if self.last_amount_of_goal_objects < len(self.goal_objects_collected_and_placed):
            self.last_amount_of_goal_objects = len(self.goal_objects_collected_and_placed)
            # mug has been picked up
            reward += 1
            print('{} reward collected! Inventory: {}'.
                  format(reward, self.goal_objects_collected_and_placed))
        elif self.last_amount_of_goal_objects > len(self.goal_objects_collected_and_placed):
            # placed mug onto/into receptacle
            pass
        self.last_amount_of_goal_objects = len(self.goal_objects_collected_and_placed)

        if self.max_episode_length and self.t >= self.max_episode_length:
            print('Reached maximum episode length: t: {}'.format(self.t))
            done = True

        return reward, done
