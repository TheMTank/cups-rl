# ai2thor-experiments

This project will focus primarily on the implementation and benchmark of different approaches to 
domain and task transfer learning in reinforcement learning. The focus lies on a diverse set of 
simplified domestic robot tasks using [ai2thor](https://ai2thor.allenai.org/), a realistic household 
3D environment. To provide an example, an agent could learn to pick up a cup under particular 
conditions and then zero/few shot transfer to pick up many different cups in many different 
situations.

We included our own wrapper for the environment as well to support the modification of the tasks 
within an openAI gym interface, so that new and more complex tasks can be developed efficiently to 
train and test the agent.

More detailed information on ai2thor environment can be found on their 
[tutorial](http://ai2thor.allenai.org/tutorials/installation).

## Overview

This project will include implementations and adaptations of the following papers as a benchmark of 
the current state of the art approaches to the problem:

- [Ikostrikov's A3C](https://github.com/ikostrikov/pytorch-a3c)
- [Gated-Attention Architectures for Task-Oriented Language Grounding](https://arxiv.org/abs/1706.07230) 
-- *Original code available on [DeepRL-Grounding](https://github.com/devendrachaplot/DeepRL-Grounding)
based on Ikostrikov's A3C* 
- [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/pdf/1710.02298.pdf) 
-- *Original code available on [Rainbow](https://github.com/Kaixhin/Rainbow) from Kaixhin*


Implementations of these can be found in the algorithms folder and can be run on [AI2ThorEnv](https://arxiv.org/pdf/1710.02298.pdf) with:  

`python algorithms/a3c/main.py`

`python algorithms/rainbow/main.py`

Check the argparse help for more details and variations of running the algorithm with different 
hyperparams and on the atari environment as well.

## Installation

Clone ai2thor-experiments repository:

```
# AI2THOR_EXPERIMENTS=/path/to/clone/ai2thor-experiments
git clone https://github.com/TheMTank/ai2thor-experiments.git $AI2THOR_EXPERIMENTS
```

Install Python dependencies (Currently only supporting python 3.5+):

`pip install -r $AI2THOR_EXPERIMENTS/requirements.txt`

Finally, add `AI2THOR_EXPERIMENTS` to your PYTHONPATH environment variable and you are done.

## How to use

The wrapper is based on OpenAI gym interfaces as described in [gym documentation](https://gym.openai.com/docs/).
Here is a simple example with the default configuration, that will place the agent in a "Kitchen" 
for the task of picking up and putting down mugs. 

```
from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
N_EPISODES = 20
env = AI2ThorEnv()
max_episode_length = env.task.max_episode_length
for episode in range(N_EPISODES):
    state = env.reset()
    for step_num in range(max_episode_length):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if done:
            break
```

### Environment and Task configurations

The environment is typically defined by a JSON configuration file located on the `gym_ai2thor/config_files` 
folder. You can find an example `config_example.json` to see how to customize it. Here there is one
as well:

```
# gym_ai2thor/config_files/myconfig.json
{'pickup_put_interaction': True,
 'open_close_interaction': true,
 'pickup_objects': ['Mug', 'Apple', 'Book'],
 'acceptable_receptacles': ['CounterTop', 'TableTop', 'Sink'],
 'openable_objects': ['Microwave'],
 'scene_id': 'FloorPlan28',
 'grayscale': True,
 'resolution': (300, 300),
 'task': {'task_name': 'PickUp',
          'target_object': 'Mug'}} 
 ```
 
For experimentation it is important to be able to make slight modifications of the environment 
 without having to create a new config file each time. The class `AI2ThorEnv` includes the keyword 
 argument `config_dict`, that allows to input a python dictionary **in addition to** the config file 
 that overrides the parameters described in the config.

The tasks are defined in `envs/tasks.py` and allow for particular configurations regarding the 
rewards given and termination conditions for an episode. You can use the tasks that we defined
there or create your own modifying the `TaskFactory` and adding it as a subclass of the `BaseTask`. 
Here an example of a new task definition:

```
# envs/tasks.py
class TaskFactory:
    ...
    elif task_name == 'MoveAheadTask':
        return MoveAheadTask(**config['task'])
    ...
```

```
# envs/tasks.py
class MoveAheadTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(kwargs)
        self.rewards = []

    def transition_reward(self, state):
        reward = 1 if state.metadata['lastAction'] == 'MoveAhead' else -1 
        self.rewards.append(reward)
        done = sum(rewards) > 100 or self.step_num > self.max_episode_length
        if done:
            self.rewards = []
        return reward, done

    def reset(self):
        self.step_num = 0
``` 

We encourage you to explore the scripts on the `examples` folder to guide you on the wrapper
 functionalities and explore how to create more customized versions of ai2thor environments and 
 tasks. 

Here is the desired result of an example task in which the goal of the agent is to place a cup in the 
microwave.

<div align="center">
  <img src="docs/cup_into_microwave.gif" width="294px" />
  <p>Example of task "place cup in microwave"</p>
</div>

## The Team

[The M Tank](http://www.themtank.org/) is a non-partisan organisation that works solely to recognise the multifaceted 
nature of Artificial Intelligence research and to highlight key developments within all sectors affected by these 
advancements. Through the creation of unique resources, the combination of ideas and their provision to the public, 
this project hopes to encourage the dialogue which is beginning to take place globally. 

To produce value for the individual, for researchers, for institutions and for the world.

## License

This project is released under the [MIT license](https://github.com/TheMTank/ai2thor-experiments/master/LICENSE).
