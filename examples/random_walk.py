import random

from envs import ThorWrapperEnv

if __name__ == '__main__':
    # Random agent example with wrapper
    env = ThorWrapperEnv()
    for episode in range(5):
        for t in range(1000):
            a = random.randint(0, len(env.ACTION_SPACE) - 1)
            s, r, done = env.step(a)
            if done:
                s = env.reset()
                break
