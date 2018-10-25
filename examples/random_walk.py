"""
Example of use for the ai2thor wrapper. It runs N_EPISODES episodes in the Environment picking
random actions.
"""
import time
import random

from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv

MAX_EPISODE_LENGTH = 5000
N_EPISODES = 20


if __name__ == '__main__':
    env = AI2ThorEnv()  # max_episode_length=MAX_EPISODE_LENGTH
    for episode in range(N_EPISODES):
        start = time.time()
        state = env.reset()
        for step_n in range(MAX_EPISODE_LENGTH):
            action = env.action_space.sample()
            state, reward, done = env.step(action)
            if done:
                break

            if step_n + 1 > 0 and (step_n + 1) % 100 == 0:
                print('Episode: {}. Step: {}/{}. Time taken: {:.3f}s'.format(episode + 1,
                                                                             step_n + 1,
                                                                             MAX_EPISODE_LENGTH,
                                                                             time.time() - start))
                start = time.time()
