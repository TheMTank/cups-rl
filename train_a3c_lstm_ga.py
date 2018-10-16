"""
Adapted from https://github.com/ikostrikov/pytorch-a3c
"""


import time

import numpy as np
# import numpy as np
import matplotlib as mpl
# mpl.use('TkAgg')  # or whatever other backend that you want. For interactive
mpl.use('Agg')  # or whatever other backend that you want. For non-interactive
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# from envs import create_atari_env
import envs
from model import ActorCritic
from a3c_lstm_ga_model import A3C_LSTM_GA
import utils

MOVEMENT_REWARD_DISCOUNT = -0.001

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train_a3c_lstm_ga(rank, args, shared_model, counter, lock, optimizer=None):
    torch.manual_seed(args.seed + rank)

    # env = envs.ThorWrapperEnv(current_object_type='Microwave', interaction=False)
    # env = envs.ThorWrapperEnv(current_object_type='Microwave', natural_language_instruction=True)
    # env = envs.ThorWrapperEnv(current_object_type='Microwave', dense_reward=True, natural_language_instruction=True)
    env = envs.ThorWrapperEnv(current_object_type='Microwave', natural_language_instruction=True, grayscale=False,
                              max_episode_length=args.max_episode_length)

    env.seed(args.seed + rank)

    # model = ActorCritic(env.observation_space.shape[0], env.action_space)
    # model = ActorCritic(1, env.action_space)

    # model = A3C_LSTM_GA(1, env.action_space).double()
    model = A3C_LSTM_GA(3, env.action_space).double()
    # model = ActorCritic(3, env.action_space)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    # state = env.reset()
    (image, instruction) = env.reset()
    instruction_idx = []
    for word in instruction.split(" "):
        instruction_idx.append(env.word_to_idx[word])
    instruction_idx = np.array(instruction_idx)

    image = torch.from_numpy(image)
    instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)

    # todo print probabilities of actions every 100 or 1000
    # todo see where the gradients flow. Feel the gradients

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

    total_length = args.total_length if args.total_length else 0
    episode_length = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if total_length > 0 and total_length % 100000 == 0:
            fn = 'checkpoint_total_length_{}.pth.tar'.format(total_length)
            utils.save_checkpoint({'total_length': total_length, 'state_dict': model.state_dict(),
                                   'optimizer': optimizer.state_dict(),
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
            tx = Variable(torch.from_numpy(np.array([episode_length])).long())
            episode_length += 1
            total_length += 1

            if env.grayscale:
                image = image.unsqueeze(0).unsqueeze(0)
            else:
                image = image.unsqueeze(0).permute(0, 3, 1, 2)
            value, logit, (hx, cx) = model((Variable(image),
                                            Variable(instruction_idx),
                                            (tx, hx, cx)))
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).data # todo check num_samples
            log_prob = log_prob.gather(1, Variable(action))

            action = action.numpy()[0, 0]
            (image, _), reward, done = env.step(action)
            # state, reward, done = env.step(action)

            reward += MOVEMENT_REWARD_DISCOUNT

            done = done or episode_length >= args.max_episode_length
            # reward = max(min(reward, 1), -1) # todo think about this

            with lock:
                counter.value += 1

            if done:
                number_of_episodes += 1
                episode_lengths.append(episode_length)
                episode_length = 0
                total_length -= 1
                (image, instruction) = env.reset()
                # todo is below needed for more sentences?
                # instruction_idx = []
                # for word in instruction.split(" "):
                #     instruction_idx.append(env.word_to_idx[word])
                # instruction_idx = np.array(instruction_idx)
                # instruction_idx = torch.from_numpy(
                #     instruction_idx).view(1, -1)

                # todo massive bug here with sums seeming wrong
                total_reward_for_episode = sum(all_rewards_in_episode)
                episode_total_rewards_list.append(total_reward_for_episode)
                all_rewards_in_episode = []

                utils.create_plots(args.experiment_id, avg_reward_for_num_steps_list, total_reward_for_num_steps_list,
                                   number_of_episodes,
                                   episode_total_rewards_list, episode_lengths, env, prob)

                print('Episode number: {}. Total minutes elapsed: {:.3f}'.format(number_of_episodes,
                                                                                 (time.time() - start) / 60.0))
                print('Total Length: {}. done after reset: {}. Total reward for episode: {}'.format(total_length, done,
                                                                                                    total_reward_for_episode))

            image = torch.from_numpy(image)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            all_rewards_in_episode.append(reward)

            if done:
                break

        print('Step no: {}. total length: {}'.format(episode_length, total_length))
        # Everything below doesn't contain interaction with the environment
        R = torch.zeros(1, 1)
        if not done:
            tx = Variable(torch.from_numpy(np.array([episode_length])).long())
            if env.grayscale:
                formatted_image = image.unsqueeze(0).unsqueeze(0)
            else:
                formatted_image = image.unsqueeze(0).permute(0, 3, 1, 2)
            value, _, _ = model((Variable(formatted_image),
                                 Variable(instruction_idx),
                                 (tx, hx, cx)))
            R = value.data

        total_reward_for_num_steps = sum(rewards)
        avg_reward_for_num_steps = total_reward_for_num_steps / len(rewards)
        total_reward_for_num_steps_list.append(total_reward_for_num_steps)
        avg_reward_for_num_steps_list.append(avg_reward_for_num_steps)

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        # import pdb;pdb.set_trace() # good place to breakpoint to see training cycle
        R = Variable(R).double()

        gae = torch.zeros(1, 1).double()
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i].double()
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * values[i + 1].data.double() - values[i].data.double()
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
