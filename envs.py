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
    # todo fix for microwave and do in one line
    if x2 < 100:
        return False
    if x1 > 200:
        return False
    if y2 < 50:
        return False
    if y1 > 200:
        return False

    return True

def close_enough(distance, less_than=1.0):
    return True if distance < less_than else False

class ThorWrapperEnv():
    def __init__(self, scene_id='FloorPlan28', task=0, max_episode_length=1000, current_object_type='Mug',
                 grayscale=True, interaction=True, dense_reward=False, natural_language_instruction=False,
                 entity_feats=True):
        self.scene_id = scene_id
        self.controller = ai2thor.controller.Controller()
        self.controller.start()

        self.controller.reset(self.scene_id)
        self.event = self.controller.step(dict(action='Initialize', gridSize=0.25,
                                               renderDepthImage=True,
                                               renderClassImage=True,
                                               renderObjectImage=True
                                               ))

        self.current_object_type = current_object_type

        self.max_episode_length = max_episode_length
        self.t = 0
        self.task = task
        self.done = False
        self.natural_language_instruction = natural_language_instruction
        if self.natural_language_instruction:
            self.train_instructions = ["Go to the microwave"]
            self.word_to_idx = self.get_word_to_idx()
        self.entity_feats = entity_feats
        self.dense_reward = dense_reward

        # action space stuff for ai2thor
        # todo loop through list so we can arbitrarily remove some actions e.g. lookup/down
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
        self.grayscale = grayscale
        self.resolution = (128, 128) #(64, 64)
        # self.resolution = (64, 64) #(64, 64)
        if self.grayscale:
            self.observation_space = np.array((1, ) + self.resolution)
        else:
            self.observation_space = np.array((3, ) + self.resolution)

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
        if self.natural_language_instruction:
            state = (self.preprocess(self.event.frame), 'Go to the microwave')
        else:
            state = self.preprocess(self.event.frame)
        if self.entity_feats:
            state = (state, self.calculate_entity_feats(self.current_object_type))

        return state, reward, self.done

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
        if self.natural_language_instruction:
            state = (self.preprocess(self.event.frame), 'Go to the microwave')
        else:
            state = self.preprocess(self.event.frame)
        if self.entity_feats:
            state = (state, self.calculate_entity_feats(self.current_object_type))
        return state

    def preprocess(self, img):
        img = skimage.transform.resize(img, self.resolution)
        img = img.astype(np.float32)
        if self.grayscale:
            img = self.rgb2gray(img)
        return img

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
                if self.dense_reward:
                    all_objects_for_object_type = [obj for obj in self.event.metadata['objects'] if
                                                   obj['objectType'] == self.current_object_type]

                    # only 1 microwave for now or first item. todo make more general!!! always choose the min distance?
                    return -all_objects_for_object_type[0]['distance']
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
        return len(self.mugs_ids_collected_and_placed) # todo double check this isn't called or change

    def is_episode_finished(self):
        if self.max_episode_length and self.t > self.max_episode_length - 1:
            return True

        if self.task == 0: # todo add check if natural language mode
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

    def calculate_entity_feats(self, object_type='Mug'):
        """
        For each entity of object type, retrieve 9 values: x, y, width, height, 3d_distance, x_3d, y_3d, z_3d, class
        """

        all_objects_for_object_type = [obj for obj in self.event.metadata['objects'] if
                                       obj['objectType'] == object_type]

        entity_feats = []
        for obj in all_objects_for_object_type:

            entity_feats.extend([obj['position']['x'], obj['position']['y'], obj['position']['z'], (obj['distance'])])
            entity_feats.append(0 if object_type == 'Mug' else 1) # todo
            # bbox feats
            bounds = self.event.instance_detections2D.get(obj['objectId'])
            if bounds is None:
                entity_feats.extend([0, 0, 0, 0])
            else:
                x1, y1, x2, y2 = bounds
                entity_feats.append(x1)
                entity_feats.append(y1)
                entity_feats.append(abs(x2 - x1))
                entity_feats.append(abs(y2 - y1))

        if len(self.event.metadata['inventoryObjects']) > 0:
            # todo can only pick up cups so this is fine but eventually can pick up anything
            # can only carry one item
            # self.event.metadata['inventoryObjects'][0]['objectId']
            # todo maybe changed 3d to agents 3d
            entity_feats.extend([0, 0, 0, 0, 0 if object_type == 'Mug' else 1,
                                 0, 0, 0, 0])

            # todo if in inventory or out of sight. And check if id changes

        return np.array(entity_feats)

    def seed(self, seed):
        return #todo

    def get_word_to_idx(self):
        word_to_idx = {}
        for instruction_data in self.train_instructions:
            instruction = instruction_data # todo actual json ['instruction']
            for word in instruction.split(" "):
                if word not in word_to_idx:
                    word_to_idx[word] = len(word_to_idx)
        return word_to_idx

if __name__ == '__main__':
    # Random agent example with wrapper
    env = ThorWrapperEnv()
    for episode in range(5):
        for t in range(1000):
            a = random.randint(0, len(env.ACTION_SPACE) - 1)
            s, r, terminal = env.step(a)
