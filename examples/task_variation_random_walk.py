"""
Example trying out different tasks within the ai2thor wrapper.
Still picks random actions but shows how much we can vary the environment.
"""
import time

from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv


if __name__ == '__main__':
    # Here we only allow apples to picked up and this our target object to pick up
    config_dict = {
        'pickup_put_interaction': True,
        'open_close_interaction': False,  # disable opening/closing objects
        'openable_objects': [],  # disable opening objects another way
        'pickup_objects': [
            "Apple"
        ],
        'scene_id': 'FloorPlan27',  # let's try a different room
        'grayscale': True,
        'resolution': [128, 128],
        'task': {
            'task_name': 'PickUpTask',
            'target_objects': {'Apple': 1}  # target object changed to Apple
        }
    }

    # Input config_dict to env which will overwrite a few values given in the default config_file.
    # Therefore, a few harmless warnings are expected
    env = AI2ThorEnv(config_dict=config_dict)
    max_episode_length = env.task.max_episode_length
    N_EPISODES = 3
    for episode in range(N_EPISODES):
        start = time.time()
        state = env.reset()
        for step_num in range(max_episode_length):
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            if done:
                break

            if step_num + 1 > 0 and (step_num + 1) % 100 == 0:
                print('Episode: {}. Step: {}/{}. Time taken: {:.3f}s'.format(episode + 1,
                                         (step_num + 1), max_episode_length, time.time() - start))
                start = time.time()
