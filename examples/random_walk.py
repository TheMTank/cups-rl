import time
import random

from ai2thor_wrapper.envs import ThorWrapperEnv

if __name__ == '__main__':
    # Random agent example with wrapper
    env = ThorWrapperEnv()
    for episode in range(20):
        start = time.time()
        s = env.reset()
        for t in range(1000):
            a = random.randint(0, len(env.ACTION_SPACE) - 1)
            s, r, done = env.step(a)
            if done:
                break

            if t + 1 > 0 and (t + 1) % 100 == 0:
                print('Episode: {}. Step: {}/{}. Time taken: {:.3f}s'.format(episode + 1, t + 1, 1000,
                                                                             time.time() - start))
                start = time.time()
