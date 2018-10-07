"""
Adapted from https://github.com/ikostrikov/pytorch-a3c
"""


import time

# import numpy as np
import matplotlib as mpl
# mpl.use('TkAgg')  # or whatever other backend that you want
mpl.use('Agg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# from envs import create_atari_env
import envs
from model import ActorCritic
import utils

MOVEMENT_REWARD_DISCOUNT = -0.001

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock, optimizer=None):
    torch.manual_seed(args.seed + rank)

    # env = envs.ThorWrapperEnv(current_object_type='Microwave', interaction=False)
    # env = envs.ThorWrapperEnv(current_object_type='Microwave', dense_reward=True)
    env = envs.ThorWrapperEnv(current_object_type='Mug', max_episode_length=args.max_episode_length)
    env.seed(args.seed + rank)
    model = ActorCritic(env.observation_space[0], env.action_space)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    # monitoring
    avg_reward_for_num_steps_list = []
    total_reward_for_num_steps_list = []
    episode_total_rewards_list = []
    all_rewards_in_episode = []
    episode_lengths = []
    number_of_episodes = 0
    start = time.time()
    # plt.ion()
    # plt.ioff()  # turn of interactive plotting mode

    total_length = 0
    episode_length = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if total_length > 0 and total_length % 100000 == 0:
            fn = 'checkpoint_total_length_{}.pth.tar'.format(total_length)
            utils.save_checkpoint({
                'total_length': total_length,
                # 'arch': args.arch,
                'state_dict': model.state_dict(),
                # 'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, args.experiment_id, fn)

        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            episode_length += 1
            total_length += 1
            value, logit, (hx, cx) = model((Variable(state.unsqueeze(0).float()),
                                            (hx, cx)))
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).data
            log_prob = log_prob.gather(1, Variable(action))

            action_int = action.numpy()[0][0].item()
            state, reward, done = env.step(action_int)

            reward += MOVEMENT_REWARD_DISCOUNT # todo should be within environment?

            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1) # todo dangerous. clamping between 1 and -1 all along?
            # todo scale/standardise all rewards

            with lock:
                counter.value += 1

            if done:
                number_of_episodes += 1
                episode_lengths.append(episode_length)
                episode_length = 0
                total_length -= 1
                state = env.reset()

                total_reward_for_episode = sum(all_rewards_in_episode)
                episode_total_rewards_list.append(total_reward_for_episode)
                all_rewards_in_episode = []

                utils.create_plots(args.experiment_id, avg_reward_for_num_steps_list, total_reward_for_num_steps_list, number_of_episodes,
                                   episode_total_rewards_list, episode_lengths)

                print('Episode number: {}. Total minutes elapsed: {:.3f}'.format(number_of_episodes,
                                                                                 (time.time() - start) / 60.0))
                print('Total Length: {}. Total reward for episode: {}'.format(total_length, done, total_reward_for_episode))

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            all_rewards_in_episode.append(reward)

            if done:
                break

        print('Step no: {}. total length: {}'.format(episode_length, total_length))
        # Everything below doesn't contain interaction with the environment
        R = torch.zeros(1, 1)
        if not done: # to change last reward to predicted value to ....
            value, _, _ = model((Variable(state.unsqueeze(0).float()), (hx, cx)))
            R = value.data

        total_reward_for_num_steps = sum(rewards)
        avg_reward_for_num_steps = total_reward_for_num_steps / len(rewards)
        total_reward_for_num_steps_list.append(total_reward_for_num_steps)
        avg_reward_for_num_steps_list.append(avg_reward_for_num_steps)

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        # import pdb;pdb.set_trace() # good place to breakpoint to see training cycle
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

        # todo checkpoint every 100k.
