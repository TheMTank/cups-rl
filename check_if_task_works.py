
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

from envs import ThorWrapperEnv

if __name__ == '__main__':
    #

    actions = [6, 6, 0, 0, 6, 0, 7, 0, 0, 0, 0, 7, 2, 8, 8, 8, 8, 1]

    env = ThorWrapperEnv(current_object_type='Microwave')
    for episode in range(2):
        for idx, a in enumerate(actions):
            # a = random.randint(0, len(env.ACTION_SPACE) - 1)
            # a = actions[t]
            # if idx > 12:
            #     import pdb;pdb.set_trace()
            s, r, terminal = env.step(a)
            print(r, terminal)
            if terminal:
                break
            # time.sleep(0.5)

        env.reset()
        time.sleep(2)
