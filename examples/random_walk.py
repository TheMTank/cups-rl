import sys
sys.path.append('..')
import time
import random

from ai2thor_wrapper.envs import ThorWrapperEnv

if __name__ == '__main__':
    # Random agent example with wrapper
    max_episode_length = 5000
    env = ThorWrapperEnv(max_episode_length=max_episode_length)
    for episode in range(20):
        start = time.time()
        s = env.reset()
        for t in range(max_episode_length):
            a = random.choice(env.ACTION_SPACE)
            s, r, done = env.step(a)
            if done:
                break

            if t + 1 > 0 and (t + 1) % 100 == 0:
                print('Episode: {}. Step: {}/{}. Time taken: {:.3f}s'.format(episode + 1, t + 1, max_episode_length,
                                                                             time.time() - start))
                start = time.time()
