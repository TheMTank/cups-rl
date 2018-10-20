# ai2thor-experiments

This project will focus primarily on the implementation and benchmark of different approaches to domain and task 
transfer learning in the frame of reinforcement learning. The focus lies on a diverse set of domestic robot tasks 
using [ai2thor](https://ai2thor.allenai.org/), a realistic household 3D environment. To provide an example, an agent 
could learn to pick up a cup under particular conditions and then zero/few shot transfer to pick up many different 
cups in many different situations.

A wrapper will be included as well to support the modification of the tasks within an openAI gym interface, so that 
new and more complex tasks can be developed efficiently to train and test the agent.

## Overview

This project will include implementations of the following papers as a benchmark of the current state of the art 
approaches to the problem:

- [Gated-Attention Architectures for Task-Oriented Language Grounding](https://arxiv.org/abs/1706.07230) -- 
*Original code available on [DeepRL-Grounding](https://github.com/devendrachaplot/DeepRL-Grounding)*

## License

This project is released under the [MIT license](https://github.com/TheMTank/ai2thor-experiments/master/LICENSE).


## Installation

Clone ai2thor-experiments repository:

```
# AI2THOR_EXPERIMENTS=/path/to/clone/ai2thor-experiments
git clone https://github.com/TheMTank/ai2thor-experiments.git $AI2THOR_EXPERIMENTS
```

Install Python dependencies:

`pip install -r $AI2THOR_EXPERIMENTS/requirements.txt`

## How to use

This section will be updated soon to show how to run examples using the provided code. In the meantime, here is an 
example task in which the goal of the agent is to place a cup in the microwave.

<div align="center">
  <img src="examples/cup_into_microwave.gif" width="700px" />
  <p>Example of task "place cup in microwave"</p>
</div>

## The Team

[The M Tank](http://www.themtank.org/) is a non-partisan organisation that works solely to recognise the multifaceted 
nature of Artificial Intelligence research and to highlight key developments within all sectors affected by these 
advancements. Through the creation of unique resources, the combination of ideas and their provision to the public, 
this project hopes to encourage the dialogue which is beginning to take place globally. 

To produce value for the individual, for researchers, for institutions and for the world.