"""
Example trying out different tasks within the ai2thor wrapper.
Still picks random actions but shows how much we can vary the environment.
"""
import time

from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv


if __name__ == '__main__':
    # Here we only allow apples to picked up and this our target object to pick up
    config_dict = {
        "env": {
            "interaction": True,
            "pickup_objects": [
                "Apple"
            ],
            "openable_objects": [],  # disable opening objects
            "scene_id": "FloorPlan27",  # we try a different room
            "grayscale": True,
            "resolution": [128, 128]
        },
        "task": {
            "task_name": "PickUp",
            "target_object": "Apple"  # target object changed to Apple
        }
    }

    # Input config_dict to env which will overwrite a few values given in the default config_file.
    # Therefore, a few warnings will occur
    env = AI2ThorEnv(config_dict=config_dict)
    max_episode_length = env.task.max_episode_length
    for episode in range(3):
        start = time.time()
        state = env.reset()
        for step_n in range(max_episode_length):
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            if done:
                break

            if step_n + 1 > 0 and (step_n + 1) % 100 == 0:
                print('Episode: {}. Step: {}/{}. Time taken: {:.3f}s'.format(episode + 1,
                                             (step_n + 1) + (episode * max_episode_length),
                                             max_episode_length, time.time() - start))
                start = time.time()
