import time
import random

import ai2thor.controller

controller = ai2thor.controller.Controller()
controller.start()

# Kitchens: FloorPlan1 - FloorPlan30
# Living rooms: FloorPlan201 - FloorPlan230
# Bedrooms: FloorPlan301 - FloorPlan330
# Bathrooms: FloorPLan401 - FloorPlan430


ACTION_SPACE = {0: dict(action='MoveAhead'),
                1: dict(action='MoveBack'),
                2: dict(action='MoveRight'),
                3: dict(action='MoveLeft'),
                4: dict(action='LookUp'),
                5: dict(action='LookDown'),
                6: dict(action='RotateRight'),
                7: dict(action='RotateLeft'),
                # 1: dict(action='OpenObject'), # needs object id
                # 1: dict(action='CloseObject'), # needs object id
                8: dict(action='PickupObject'), # needs object id???
                9: dict(action='PutObject') # needs object id
                }

# also Teleport and TeleportFull but obviously only used for initialisation
NUM_ACTIONS = len(ACTION_SPACE.keys())

controller.reset('FloorPlan28')
event = controller.step(dict(action='Initialize', gridSize=0.25,
                     renderDepthImage=True,
                     renderClassImage=True,
                     renderObjectImage=True))

for t in range(10000):
    random_a_space_int = random.randint(0, NUM_ACTIONS - 1)
    if random_a_space_int == 8:
        if len(event.metadata['inventoryObjects']) == 0:
            for o in event.metadata['objects']:
                if o['visible'] and (o['objectType'] == 'Mug'):
                    event = controller.step(
                        dict(action='PickupObject', objectId=o['objectId']), raise_for_failure=True)
                    break
            continue
    elif random_a_space_int == 9:
        # action = dict(action='PutObject', )
        if len(event.metadata['inventoryObjects']) > 0:

            for o in event.metadata['objects']:
                if o['visible'] and (o['objectType'] == 'CounterTop' or
                                     o['objectType'] == 'TableTop' or
                                     o['objectType'] == 'Sink' or
                                     o['objectType'] == 'CoffeeMachine' or
                                     o['objectType'] == 'Box'):
                    import pdb;pdb.set_trace()
                    event = controller.step(dict(action='PutObject', objectId=event.metadata['inventoryObjects'][0]['objectId'], receptacleObjectId=o['objectId']), raise_for_failure=True)
                    break
        continue
    else:
        action = ACTION_SPACE[random_a_space_int]
        # if random_a_space_int == 8:
        #     time.sleep(5)
        event = controller.step(action)
        if len(event.metadata['inventoryObjects']) > 0:
            print(event.metadata['inventoryObjects'])
        # time.sleep(0.5)
