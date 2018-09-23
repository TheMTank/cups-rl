"""
Copied and modified from
http://ai2thor.allenai.org/tutorials/examples
"""

import time
from pprint import pprint

import ai2thor.controller
controller = ai2thor.controller.Controller()
controller.start()

# Kitchens: FloorPlan1 - FloorPlan30
# Living rooms: FloorPlan201 - FloorPlan230
# Bedrooms: FloorPlan301 - FloorPlan330
# Bathrooms: FloorPLan401 - FloorPlan430

controller.reset('FloorPlan28')
event = controller.step(dict(action='Initialize', gridSize=0.25,
                                 renderDepthImage=True,
                                 renderClassImage=True,
                                 renderObjectImage=True)) # todo these do nothing why????

event = controller.step(dict(action='MoveAhead'))

# Numpy Array - shape (width, height, channels), channels are in RGB order
print(event.frame)
print(event.frame.shape)

# Numpy Array in BGR order suitable for use with OpenCV
event.cv2image()

# current metadata dictionary that includes the state of the scene
print(event.metadata)#
pprint(event.metadata)

controller.step(dict(action='RotateRight'))
controller.step(dict(action='RotateRight'))
event = controller.step(dict(action='MoveAhead'))
event = controller.step(dict(action='MoveAhead'))
controller.step(dict(action='RotateRight'))
event = controller.step(dict(action='MoveAhead'))
event = controller.step(dict(action='MoveAhead'))
controller.step(dict(action='RotateLeft'))
event = controller.step(dict(action='MoveAhead'))
event = controller.step(dict(action='MoveAhead'))

def get_distance_to_coffee_cups(event):
    mugs = [obj for obj in event.metadata['objects'] if obj['objectType'] == 'Mug']

    return mugs

# all_object_names = [obj['name'] for obj in event.metadata['objects']]
# print(all_object_names)

mugs = get_distance_to_coffee_cups(event)
pprint(mugs)



time.sleep(1)
# print(event)

