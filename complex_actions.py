"""
Copied and modified from
http://ai2thor.allenai.org/tutorials/examples
"""

import time
import ai2thor.controller
controller = ai2thor.controller.Controller()
controller.start()

default_wait_time = 2
controller.reset('FloorPlan28')
# time.sleep(3)
controller.step(dict(action='Initialize', gridSize=0.25))

# time.sleep(3)
controller.step(dict(action='Teleport', x=-1.25, y=1.00, z=-1.5))
time.sleep(3)
controller.step(dict(action='LookDown'))
time.sleep(default_wait_time)
event = controller.step(dict(action='Rotate', rotation=90))
time.sleep(default_wait_time)
all_object_names = [obj['name'] for obj in event.metadata['objects']]
print(all_object_names)
# import pdb;pdb.set_trace()
# In FloorPlan28, the agent should now be looking at a mug
for o in event.metadata['objects']:
    if o['visible'] and o['pickupable'] and o['objectType'] == 'Mug':
        event = controller.step(dict(action='PickupObject', objectId=o['objectId']), raise_for_failure=True)
        mug_object_id = o['objectId']
        break
time.sleep(default_wait_time)
all_object_names = [obj['name'] for obj in event.metadata['objects']]
print(all_object_names)
# the agent now has the Mug in its inventory
# to put it into the Microwave, we need to open the microwave first

event = controller.step(dict(action='LookUp'))
time.sleep(default_wait_time)
for o in event.metadata['objects']:
    if o['visible'] and o['openable'] and o['objectType'] == 'Microwave':
        event = controller.step(dict(action='OpenObject', objectId=o['objectId']), raise_for_failure=True)
        receptacle_object_id = o['objectId']
        time.sleep(default_wait_time)
        break

event = controller.step(dict(action='MoveRight'), raise_for_failure=True)
time.sleep(default_wait_time)
event = controller.step(dict(
    action='PutObject',
    receptacleObjectId=receptacle_object_id,
    objectId=mug_object_id), raise_for_failure=True)
time.sleep(default_wait_time)

# close the microwave
event = controller.step(dict(
    action='CloseObject',
    objectId=receptacle_object_id), raise_for_failure=True)
time.sleep(default_wait_time)
