"""
Auxiliary functions for building environments
"""
import os
import json
import warnings

from gym import error


def read_config(config_path, config_dict=None):
    """
    Returns the parsed information from the config file and specific info can be passed or
    overwritten with the config_dict. Full example below:

    {
        "env": {
            "interaction": true,
            "pickup_objects": [
                "Mug",
                "Apple",
                "Book"
            ],
            "acceptable_receptacles": [
                "CounterTop",
                "TableTop",
                "Sink"
            ],
            "openable_objects": [
                "Microwave"
            ],
            "scene_id": "FloorPlan28",
            "grayscale": true,
            "resolution": [128, 128]
        },
        "task": {
            "task_name": "PickUp",
            "target_object": "Mug"
        }
    }

    """
    config_path = os.path.join(os.path.dirname(__file__), config_path)
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        raise error.Error('No config file found at: {}. Exiting'.format(config_path))

    if config_dict:
        high_level_keys = [x for x in ['env', 'task'] if x in config_dict]
        if not high_level_keys:
            raise error.Error('keys "env" or "task" were not found in config_dict'.
                              format(config_path))
        for high_level_key in high_level_keys:
            for key in config_dict[high_level_key]:
                if key in config[high_level_key]:
                    warnings.warn('Key: {} already in config file for {}. Overwriting with value: '
                                  '{}'.format(key, high_level_key, config_dict[high_level_key][key]))
                config[high_level_key][key] = config_dict[high_level_key][key]

    return config

class InvalidTaskParams(Exception):
    """
    Raised when the user inputs the wrong parameters for creating a task.
    """
    pass
