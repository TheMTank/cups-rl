import random

import cv2
import numpy as np
import skimage.color, skimage.transform
import ai2thor.controller

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

class ThorWrapperEnv():
    def __init__(self, scene_id='FloorPlan28', task=0, max_episode_length=1000, current_object_type='Mug', interaction=True):
        self.scene_id = scene_id
        self.controller = ai2thor.controller.Controller()
        self.controller.start()

        self.controller.reset(self.scene_id)
        self.event = self.controller.step(dict(action='Initialize', gridSize=0.25,
                                               renderDepthImage=True,
                                               renderClassImage=True,
                                               renderObjectImage=True
                                               ))

        # self.current_object_type = 'Mug'
        # self.current_object_type = 'Microwave'
        self.current_object_type = current_object_type

        self.max_episode_length = max_episode_length
        self.t = 0
        self.task = task
        self.done = False

        # action space stuff for ai2thor
        self.ACTION_SPACE = {0: dict(action='MoveAhead'),
                            1: dict(action='MoveBack'),
                            2: dict(action='MoveRight'),
                            3: dict(action='MoveLeft'),
                            4: dict(action='LookUp'),
                            5: dict(action='LookDown'),
                            6: dict(action='RotateRight'),
                            7: dict(action='RotateLeft'),
                            # 1: dict(action='OpenObject'), # needs object id
                            # 1: dict(action='CloseObject'), # needs object id
                            8: dict(action='PickupObject'),  # needs object id???
                            9: dict(action='PutObject')  # needs object id
                            }

        if not interaction:
            self.ACTION_SPACE.pop(8)
            self.ACTION_SPACE.pop(9)

        # also Teleport and TeleportFull but obviously only used for initialisation
        self.NUM_ACTIONS = len(self.ACTION_SPACE.keys())
        self.action_space = self.NUM_ACTIONS
        self.resolution = (64, 64)
        self.observation_space = np.array((1, ) + self.resolution)

        self.mugs_ids_collected_and_placed = set()
        self.last_amount_of_mugs = len(self.mugs_ids_collected_and_placed)

    def step(self, action_int):
        if action_int == 8: # 8: dict(action='PickupObject')
            if len(self.event.metadata['inventoryObjects']) == 0:
                for o in self.event.metadata['objects']:
                    if o['visible'] and (o['objectType'] == 'Mug'):
                        mug_id = o['objectId']
                        self.event = self.controller.step(
                            dict(action='PickupObject', objectId=mug_id), raise_for_failure=True)
                        self.mugs_ids_collected_and_placed.add(mug_id)
                        print(self.mugs_ids_collected_and_placed, self.event.metadata['inventoryObjects'])
                        break
        elif action_int == 9: # action = dict(action='PutObject', )
            if len(self.event.metadata['inventoryObjects']) > 0:

                for o in self.event.metadata['objects']:
                    if o['visible'] and o['receptacle'] and (o['objectType'] == 'CounterTop' or
                                                             o['objectType'] == 'TableTop' or
                                                             o['objectType'] == 'Sink' or
                                                             o['objectType'] == 'CoffeeMachine' or
                                                             o['objectType'] == 'Box'):
                        mug_id = self.event.metadata['inventoryObjects'][0]['objectId']
                        try:
                            self.event = self.controller.step(dict(action='PutObject', objectId=mug_id, receptacleObjectId=o['objectId']),
                                                    raise_for_failure=True)
                            self.mugs_ids_collected_and_placed.remove(mug_id)
                            print(self.mugs_ids_collected_and_placed, self.event.metadata['inventoryObjects'])
                        except Exception as e:
                            # sometimes crashes here for placing mug onto table top which should be fine except distance?
                            # import pdb;pdb.set_trace()
                            print(e)
                            test = 5
                        # reward = self.calculate_reward(mug_id)
                        break
        else:
            action = self.ACTION_SPACE[action_int]
            self.event = self.controller.step(action)

        self.t += 1
        self.done = self.is_episode_finished()
        reward = self.calculate_reward(self.done)
        return self.preprocess(self.event.frame), reward, self.done

    def reset(self):
        self.t = 0
        self.controller.reset(self.scene_id)
        # todo check to see if inventory properly reset
        self.event = self.controller.step(dict(action='Initialize', gridSize=0.25, renderDepthImage=True,
                                               renderClassImage=True,
                                               renderObjectImage=True))
        self.mugs_ids_collected_and_placed = set()
        self.last_amount_of_mugs = len(self.mugs_ids_collected_and_placed)
        self.done = False
        print('Just resetted. Current self.event.metadata["inventory"]: {}'.format(self.event.metadata['inventoryObjects']))
        return self.preprocess(self.event.frame)

    def preprocess(self, img):
        img = skimage.transform.resize(img, self.resolution)
        img = img.astype(np.float32)
        # return img
        gray = self.rgb2gray(img)
        return gray

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def calculate_reward(self, done=False):
        # todo create task interface or even class for then specifying easily thousands of tasks
        if self.task == 0:
            # Go to current object and focus on it
            if done:
                num_objects_in_view_and_close = self.check_if_focus_and_close_enough_to_object_type(self.current_object_type)
                if num_objects_in_view_and_close > 0:
                    print('Stared at object and is close enough. Num objects in view and close: {}'.format(num_objects_in_view_and_close))
                    return 20
                else:
                    return -5
            else:
                return 0
        elif self.task == 1:
            if done:
                return 20
            if self.last_amount_of_mugs != len(self.mugs_ids_collected_and_placed):
                if self.last_amount_of_mugs < len(self.mugs_ids_collected_and_placed):
                    self.last_amount_of_mugs = len(self.mugs_ids_collected_and_placed)
                    # has correctly picked up cup if we are here
                    print('Reward collected!!!!!! {}'.format(self.mugs_ids_collected_and_placed))
                    return 1
                elif self.last_amount_of_mugs > len(self.mugs_ids_collected_and_placed):
                    # placed cup
                    pass
            self.last_amount_of_mugs = len(self.mugs_ids_collected_and_placed)
            return 0
        elif self.task == 2:
            pass # todo only if all 3 mugs are collected
            # if mug_id in mugs_ids_collected_and_placed:
            #     # already collected
            #     return 0
            # else:
            #     mugs_ids_collected_and_placed.add(mug_id)
            #     print('Reward collected!!!!!! {}'.format(mugs_ids_collected_and_placed))
            #     return 1.0


    def get_total_reward(self):
        return len(self.mugs_ids_collected_and_placed)

    def is_episode_finished(self):
        if self.max_episode_length and self.t > self.max_episode_length:
            return True

        if self.task == 0:
            return True if self.check_if_focus_and_close_enough_to_object_type(self.current_object_type) > 0 else False
        else:
            if len(self.mugs_ids_collected_and_placed) == 3:
                # todo this is called before the total reward
                self.mugs_ids_collected_and_placed = set()
                return True
            else:
                return False

    def is_episode_termination_success(self):
        pass

    def check_if_focus_and_close_enough_to_object_type(self, object_type='Mug'):
        all_objects_for_object_type = [obj for obj in self.event.metadata['objects'] if obj['objectType'] == object_type]

        bool_list = []
        for idx, obj in enumerate(all_objects_for_object_type):
            bounds = self.event.instance_detections2D.get(obj['objectId'])
            if bounds is None:
                continue

            x1, y1, x2, y2 = bounds
            bool_list.append(check_if_focus_and_close_enough(x1, y1, x2, y2, obj['distance']))

        return sum(bool_list)

    def seed(self, seed):
        return #todo

if __name__ == '__main__':
    # Random agent example with wrapper
    env = ThorWrapperEnv()
    for episode in range(5):
        for t in range(1000):
            a = random.randint(0, len(env.ACTION_SPACE) - 1)
            s, r, terminal = env.step(a)
