import random
import time
import unittest

from ai2thor_wrapper.envs import ThorWrapperEnv

class TestAI2ThorWrapperEnv(unittest.TestCase):
    def test_environments_runs_and_check_speed(self):
        """
        Checks to see if the environment still runs and nothing breaks. Useful for continuous deployment and keeping
        master stable. Also, we check how much time 100 steps takes in the environment takes. Final assert
        checks if max_episode_length is equal to the number of steps taken and no off-by-one errors.
        """

        num_steps = 10
        env = ThorWrapperEnv(max_episode_length=num_steps)
        start = time.time()
        all_step_times = []
        s = env.reset()
        for t in range(num_steps):
            start_of_step = time.time()
            a = random.randint(0, len(env.ACTION_SPACE) - 1)
            s, r, done = env.step(a)
            t += 1

            time_for_step = time.time() - start_of_step
            print('Step: {}. env.t: {}. Time taken for step: {:.3f}'.format(t, env.t, time_for_step))
            all_step_times.append(time_for_step)

            if done:
                break

        print('Time taken altogether: {}\nAverage time taken per step: {:.3f}'.format(
                            time.time() - start, sum(all_step_times) / len(all_step_times)))

        self.assertTrue(len(all_step_times) == num_steps)

if __name__ == '__main__':
    unittest.main()
