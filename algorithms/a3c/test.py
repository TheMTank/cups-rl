"""
Adapted from https://github.com/ikostrikov/pytorch-a3c/blob/master/test.py

Contains the testing loop of the shared model within A3C (no optimisation/backprop needed)
Usually this is run concurrently while training occurs and is useful for tracking progress. But to
save resources we can choose to only test every args.test_sleep_time seconds.
"""

import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from gym_ai2thor.task_utils import turn_instruction_str_to_tensor
from algorithms.a3c.envs import create_atari_env
from algorithms.a3c.model import ActorCritic, A3C_LSTM_GA


def test(rank, args, shared_model, counter):
    torch.manual_seed(args.seed + rank)

    if args.atari:
        env = create_atari_env(args.atari_env_name)
    else:
        env = AI2ThorEnv(config_dict=args.config_dict)
    env.seed(args.seed + rank)

    if args.atari:
        env = create_atari_env(args.atari_env_name)
    else:
        env = AI2ThorEnv(config_dict=args.config_dict)
    env.seed(args.seed + rank)

    if env.task.task_has_language_instructions:
        model = A3C_LSTM_GA(env.observation_space.shape[0], env.action_space.n,
                                   args.frame_dim, len(env.task.word_to_idx),
                                   args.max_episode_length)
    else:
        model = ActorCritic(env.observation_space.shape[0], env.action_space.n, args.frame_dim)

    model.eval()

    state = env.reset()
    if not env.task.task_has_language_instructions:
        image = torch.from_numpy(state)
    else:
        # natural language instruction is within state so unpack tuple
        (image, instruction) = state
        image = torch.from_numpy(image)
        instruction_indices = turn_instruction_str_to_tensor(instruction, env)
    done = True

    start_time = time.time()
    reward_sum = 0

    # a quick hack to prevent the agent from getting stuck
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        if args.atari and args.atari_render:
            env.render()
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            if not env.task.task_has_language_instructions:
                value, logit, (hx, cx) = model((image.unsqueeze(0).float(), (hx, cx)))
            else:
                tx = torch.from_numpy(np.array([episode_length])).long()
                value, logit, (hx, cx) = model((image.unsqueeze(0).float(),
                                                instruction_indices.long(),
                                                (tx, hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        # i.e. in test mode an agent can repeat an action ad infinitum
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            print('In test. Episode over because agent repeated action {} times'.format(
                                                                                actions.maxlen))
            done = True

        if done:
            print("Time {}, num steps over all threads {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            if env.task.task_has_language_instructions:
                (image, instruction) = state
                instruction_indices = turn_instruction_str_to_tensor(instruction, env)
            time.sleep(args.test_sleep_time)

        if not env.task.task_has_language_instructions:
            image = torch.from_numpy(state)
        else:
            (image, instruction) = state
            image = torch.from_numpy(image)
            instruction_indices = turn_instruction_str_to_tensor(instruction, env)
