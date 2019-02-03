"""
Auxiliary functions for building environments
"""
import os
import json
import math
import warnings

from gym import error


def read_config(config_path, config_dict=None):
    """
    Returns the parsed information from the config file and specific info can be passed or
    overwritten with the config_dict. Full example below:

    {
        "pickup_put_interaction": true,
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
        "resolution": [128, 128],
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
        for key in config_dict:
            # if key is task, need to loop through inner task obj and check for overwrites
            if key == 'task':
                for task_key in config_dict[key]:
                    if task_key in config[key]:
                        warnings.warn('Key: [\'{}\'][\'{}\'] already in config file with value {}. '
                                      'Overwriting with value: {}'.format(key, task_key,
                                                config[key][task_key], config_dict[key][task_key]))
                        config[key][task_key] = config_dict[key][task_key]
            # else just a regular check
            elif key in config:
                warnings.warn('Key: {} already in config file with value {}. '
                              'Overwriting with value: {}'.format(key, config[key],
                                                                  config_dict[key]))
            config[key] = config_dict[key]

    return config

class InvalidTaskParams(Exception):
    """
    Raised when the user inputs the wrong parameters for creating a task.
    """
    pass

def check_if_focus_and_close_enough_to_object_type(event, object_type='Mug'):
    all_objects_for_object_type = [obj for obj in event.metadata['objects']
                                   if obj['objectType'] == object_type]

    assert len(all_objects_for_object_type) > 0  # todo check if fails

    bool_list = []
    for idx, obj in enumerate(all_objects_for_object_type):
        bounds = event.instance_detections2D.get(obj['objectId'])
        if bounds is None:
            continue

        x1, y1, x2, y2 = bounds
        a_x, a_y, a_z = event.metadata['agent']['position']['x'], \
                        event.metadata['agent']['position']['y'], \
                        event.metadata['agent']['position']['z']
        obj_x, obj_y, obj_z = obj['position']['x'], obj['position']['y'], obj['position']['z']
        euclidean_distance_to_obj = math.sqrt((obj_x - a_x) ** 2 + (obj_y - a_y) ** 2 +
                                              (obj_z - a_z) ** 2)
        bool_list.append(check_if_focus_and_close_enough(x1, y1, x2, y2, euclidean_distance_to_obj))

    return sum(bool_list)

def check_if_focus_and_close_enough(x1, y1, x2, y2, distance):
    focus_bool = is_bounding_box_centre_close_to_crosshair(x1, y1, x2, y2)
    close_bool = euclidean_close_enough(distance)

    return True if focus_bool and close_bool else False

def is_bounding_box_centre_close_to_crosshair(x1, y1, x2, y2, threshold_within=100):
    """
    object's bounding box has to be mostly within the 100x100 middle of the image
    """
    bbox_x_cent, bbox_y_cent = (x2 + x1) / 2, (y2 + y1) / 2
    dist = math.sqrt((150 - bbox_x_cent) ** 2 + (150 - bbox_y_cent) ** 2)
    return True if dist < threshold_within else False

def euclidean_close_enough(distance, threshold=1.0):
    return True if distance < threshold else False
