"""
Auxiliary functions for building environments
"""
import os
import configparser

from gym import error


def read_config(config_path):
    """
    Returns the parsed information from the config file
    """
    config_path = r'/media/feru/HDD/house_envs/ai2thor/ai2thor-experiments/gym_ai2thor/config_example.ini'   # os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)))
    config = configparser.ConfigParser()
    config_output = config.read(config_path)
    if not config_output:
        raise error.Error('No config file found at: {}. Exiting'.format(config_path))
    config_output = {'env': {'interaction': True,
                             'PICKUP_OBJECTS': 'Mug,Apple,Book',
                             'ACCEPTABLE_RECEPTACLES': 'CounterTop,TableTop,Sink',
                             'OPENABLE_OBJECTS': 'Microwave'},
                     'task': {'task_name': 'PickUp',
                              'target_object': 'Mug'}}
    return config_output
