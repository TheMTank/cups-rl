import math

##############################
#----- Preprocessing -----#
##############################

def get_word_to_idx(train_instructions):
    word_to_idx = {}
    for instruction_data in train_instructions:
        instruction = instruction_data  # todo actual json ['instruction']
        for word in instruction.split(" "):
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
    return word_to_idx

##############################
#----- Reward Functions -----#
##############################

def check_if_focus_and_close_enough_to_object_type(event, object_type='Mug', distance_threshold=1.0):
    all_objects_for_object_type = [obj for obj in event.metadata['objects']
                                   if obj['objectType'] == object_type]

    assert len(all_objects_for_object_type) > 0  # todo check if fails ever

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
        bool_list.append(check_if_focus_and_close_enough(x1, y1, x2, y2, euclidean_distance_to_obj,
                                                         distance_threshold))

    return sum(bool_list)

def check_if_focus_and_close_enough(x1, y1, x2, y2, distance, distance_threshold):
    focus_bool = is_bounding_box_centre_close_to_crosshair(x1, y1, x2, y2)
    close_bool = euclidean_close_enough(distance, distance_threshold)

    return True if focus_bool and close_bool else False

def is_bounding_box_centre_close_to_crosshair(x1, y1, x2, y2, threshold_within=100):
    """
    object's bounding box has to be mostly within the 100x100 middle of the image
    """
    bbox_x_cent, bbox_y_cent = (x2 + x1) / 2, (y2 + y1) / 2
    dist = math.sqrt((150 - bbox_x_cent) ** 2 + (150 - bbox_y_cent) ** 2)
    return True if dist < threshold_within else False

def euclidean_close_enough(distance, threshold):
    return True if distance < threshold else False
