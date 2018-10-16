"""
Copied and modified from
http://ai2thor.allenai.org/tutorials/examples
"""

import time
from pprint import pprint
import random

import numpy as np
import skimage.color, skimage.transform
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
#mpl.use('Agg')  # or whatever other backend that you want
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import ai2thor.controller

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def show_bounding_boxes_old(key_terms=None):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    show_all = True if not key_terms else False

    check_if = lambda x, name: True if x.lower() in name.lower() else False

    mapping_color_to_name = {tuple(x['color']): x['name'] for x in event.metadata['colors']}
    mapping_color_to_bounds = {tuple(x['color']): x['bounds'] for x in event.metadata['colorBounds']}
    color_bound_names = [{'color': d['color'],
                          'name': mapping_color_to_name[tuple(d['color'])],
                          'bounds': mapping_color_to_bounds[tuple(d['color'])]} for d in event.metadata['colorBounds']]

    plt.imshow(event.frame, interpolation='nearest')
    for c_b_n in color_bound_names:
        if show_all or any([check_if(term, c_b_n['name']) for term in key_terms]):
            x1, y1, x2, y2 = c_b_n['bounds']
            # todo convert rgb color to hex and put below
            print(c_b_n['name'], c_b_n['color'], c_b_n['color'])
            print(x1, y1, x2, y2)

            # rectangle y-axis is top to bottom so invert, ai2thor rects begin bottom left, matplotlib expects top left
            rect = patches.Rectangle((x1, 300 - y1 - abs(y2 - y1)), abs(x2 - x1), abs(y2 - y1), linewidth=1,
                                     edgecolor=matplotlib_colors[random.randint(0, len(matplotlib_colors
                                                                                       ) - 1)], facecolor='none')
            ax2.add_patch(rect)
    plt.show()

def show_bounding_boxes(key_terms=None):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    show_all = True if not key_terms else False

    check_if = lambda x, name: True if x.lower() in name.lower() else False

    plt.imshow(event.frame, interpolation='nearest')
    for key, arr in event.instance_detections2D.items():
        if show_all or any([check_if(term, key) for term in key_terms]):
            x1, y1, x2, y2 = list(arr) #c_b_n['bounds']
            # todo convert rgb color to hex and put below
            print(key)
            print(x1, y1, x2, y2)

            rect = patches.Rectangle((x1, y1), abs(x2 - x1), abs(y2 - y1), linewidth=1,
                                     edgecolor=matplotlib_colors[random.randint(0, len(matplotlib_colors) - 1)],
                                     facecolor='none')
            ax2.add_patch(rect)
    plt.show()

def check_if_focus_and_close_enough_to_object_type_old(object_type='Mug'):
    all_objects_for_object_type = [obj for obj in event.metadata['objects'] if obj['objectType'] == object_type]
    # distances_to_obj_type = [x['distance'] for x in all_objects_for_object_type]

    mapping_color_to_name = {tuple(x['color']): x['name'] for x in event.metadata['colors']}
    mapping_color_to_bounds = {tuple(x['color']): x['bounds'] for x in event.metadata['colorBounds']}
    color_bound_names = [{'color': d['color'],
                          'name': mapping_color_to_name[tuple(d['color'])],
                          'bounds': mapping_color_to_bounds[tuple(d['color'])]} for d in event.metadata['colorBounds']]

    bool_list = []
    # for idx, c_b_n in enumerate(color_bound_names):
    for idx, obj in enumerate(all_objects_for_object_type):
        c_b_n_with_same_name_as_obj_id = [x for x in color_bound_names if x['name'] == obj['objectId']]
        if len(c_b_n_with_same_name_as_obj_id) == 0:
            continue
        assert len(c_b_n_with_same_name_as_obj_id) == 1

        x1, y1, x2, y2 = c_b_n_with_same_name_as_obj_id[0]['bounds']

        # check_if_focus_and_close_enough(x1, y1, x2, y2, distances_to_obj_type[idx])
        bool_list.append(check_if_focus_and_close_enough(x1, y1, x2, y2, obj['distance']))

    return sum(bool_list)

def check_if_focus_and_close_enough_to_object_type(object_type='Mug'):
    all_objects_for_object_type = [obj for obj in event.metadata['objects'] if obj['objectType'] == object_type]

    bool_list = []
    for idx, obj in enumerate(all_objects_for_object_type):
        bounds = event.instance_detections2D.get(obj['objectId'])
        if bounds is None:
            continue

        x1, y1, x2, y2 = bounds
        bool_list.append(check_if_focus_and_close_enough(x1, y1, x2, y2, obj['distance']))

    return sum(bool_list)


def check_if_focus_and_close_enough(x1, y1, x2, y2, distance):
    focus_bool = is_bounding_box_close_to_crosshair(x1, y1, x2, y2)
    close_bool = close_enough(distance)

    return True if focus_bool and close_bool else False

def is_bounding_box_close_to_crosshair(x1, y1, x2, y2):
    """
        object's bounding box has to be mostly within the 100x100 middle of the image
    """

    if x2 < 100:
        return False
    if x1 > 200:
        return False
    if y2 < 50:
        return False
    if y1 > 200:
        return False

    return True

def close_enough(distance):
    if distance < 1.0:
        return True
    return False

if __name__ == '__main__':
    # Kitchens: FloorPlan1 - FloorPlan30
    # Living rooms: FloorPlan201 - FloorPlan230
    # Bedrooms: FloorPlan301 - FloorPlan330
    # Bathrooms: FloorPLan401 - FloorPlan430

    controller = ai2thor.controller.Controller()
    controller.start()
    controller.reset('FloorPlan28')
    event = controller.step(dict(action='Initialize', gridSize=0.25,
                                     renderDepthImage=True,
                                     renderClassImage=True,
                                     renderObjectImage=True))

    # Numpy Array - shape (width, height, channels), channels are in RGB order
    print(event.frame)
    print(event.frame.shape)

    # event = controller.step(dict(action='MoveAhead'))
    event = controller.step(dict(action='RotateRight'))
    event = controller.step(dict(action='RotateRight'))
    event = controller.step(dict(action='MoveAhead'))
    event = controller.step(dict(action='MoveAhead'))
    event = controller.step(dict(action='RotateRight'))
    event = controller.step(dict(action='MoveAhead'))
    event = controller.step(dict(action='RotateLeft'))
    event = controller.step(dict(action='MoveAhead'))
    event = controller.step(dict(action='MoveAhead'))
    event = controller.step(dict(action='MoveAhead'))
    event = controller.step(dict(action='MoveAhead'))
    all_object_names = [obj['name'] for obj in event.metadata['objects']]
    visible_objects = [obj for obj in event.metadata['objects'] if obj['visible']]

    matplotlib_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


    # 98 75 126 109 - mug a bit to left middle centre
    # 165 75 194 109 - mug a bit to middle right bottom
    show_bounding_boxes()
    show_bounding_boxes(['mug', 'cup'])
    reward = check_if_focus_and_close_enough_to_object_type()
    #
    # # Show preprocessed image
    resolution = (128, 128)
    img = skimage.transform.resize(event.frame, resolution)
    plt.imshow(img) # show colour pre-processed (works in 0-1 range)
    plt.show()
    img = img.astype(np.float32)
    gray = rgb2gray(img)
    gray_unsqueezed = np.expand_dims(gray, 0)
    gray_3_channel = np.concatenate([gray_unsqueezed, gray_unsqueezed, gray_unsqueezed])
    gray_3_channel = np.moveaxis(gray_3_channel, 0, 2)
    plt.imshow(gray_3_channel)
    plt.show()

    # Can walk and step through environment interactively by copying commands and deciding when to show bounding boxes
    import pdb;pdb.set_trace()
