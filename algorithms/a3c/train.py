"""
Adapted from https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py

Contains the train code run by each A3C process on either Atari or AI2ThorEnv.
For initialisation, we set up the environment, seeds, shared model and optimizer.
In the main training loop, we always ensure the weights of the current model are equal to the
shared model. Then the algorithm interacts with the environment args.num_steps at a time,
i.e it sends an action to the env for each state and stores predicted values, rewards, log probs
and entropies to be used for loss calculation and backpropagation.
After args.num_steps has passed, we calculate advantages, value losses and policy losses using
Generalized Advantage Estimation (GAE) with the entropy loss added onto policy loss to encourage
exploration. Once these losses have been calculated, we add them all together, backprop to find all
gradients and then optimise with Adam and we go back to the start of the main training loop.

if natural_language is set to True, environment returns sentence instruction with image as state.
A3C_LSTM_GA model is used instead.
"""

import time
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from algorithms.a3c.envs import create_atari_env
from algorithms.a3c.model import ActorCritic, A3C_LSTM_GA

def save_checkpoint(state, checkpoint_path, filename, is_best=False):
    fp = os.path.join(checkpoint_path, filename)
    torch.save(state, fp)
    print('Saved model to path: {}'.format(fp))
    # if is_best:  # todo keep
    #     shutil.copyfile(filepath, 'model_best.pth.tar')

def turn_instruction_str_to_tensor(instruction, env):
    instruction_indices = []
    for word in instruction.split(" "):
        instruction_indices.append(env.task.word_to_idx[word])
        instruction_indices = np.array(instruction_indices)
        instruction_indices = torch.from_numpy(instruction_indices).view(1, -1)
    return instruction_indices

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock, optimizer=None):
    """
    Main A3C train loop and initialisation
    """
    torch.manual_seed(args.seed + rank)

    if args.atari:
        env = create_atari_env(args.atari_env_name)
    else:
        env = AI2ThorEnv(config_dict=args.config_dict)
    env.seed(args.seed + rank)

    if args.natural_language:
        model = A3C_LSTM_GA(env.observation_space.shape[0], env.action_space.n, args.frame_dim)
    else:
        model = ActorCritic(env.observation_space.shape[0], env.action_space.n, args.frame_dim)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    if not args.natural_language:
        image = torch.from_numpy(state)
    else:
        # natural language instruction is within state so unpack tuple
        (image, instruction) = state
        image = torch.from_numpy(image)
        instruction_indices = turn_instruction_str_to_tensor(instruction, env)
    done = True

    # monitoring
    total_reward_for_num_steps_list = []
    episode_total_rewards_list = []
    all_rewards_in_episode = []
    avg_reward_for_num_steps_list = []
    episode_lengths = []
    p_losses = []
    v_losses = []

    start = time.time()
    total_length = args.total_length if args.total_length else 0
    episode_length = 0
    num_backprops = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        interaction_start_time = time.time()
        for step in range(args.num_steps):
            if rank == 0 and total_length > 0 and total_length % (100000 // args.num_processes) == 0:
                # todo make function/clearer
                fn = 'checkpoint_total_length_{}.pth.tar'.format(total_length)
                checkpoint_dict = {
                    'total_length': total_length,
                    'episode_number': len(episode_lengths),
                    'counter': counter.value,
                    # todo save more stuff Need to save lists of all rewards and other info
                    # todo do it nicely and in a more extensible way?
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                save_checkpoint(checkpoint_dict, args.checkpoint_path, fn)

            episode_length += 1
            total_length += 1
            '''
            # todo check if needed
            if env.grayscale:
                image = image.unsqueeze(0).unsqueeze(0)
            else:
                image = image.unsqueeze(0).permute(0, 3, 1, 2)'''
            if not args.natural_language:
                value, logit, (hx, cx) = model((image.unsqueeze(0).float(), (hx, cx)))
            else:
                tx = torch.from_numpy(np.array([episode_length])).long()
                value, logit, (hx, cx) = model((image.unsqueeze(0).float(),
                                                instruction_indices.long(),
                                                (tx, hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            action_int = action.numpy()[0][0].item()
            state, reward, done, _ = env.step(action_int)

            done = done or episode_length >= args.max_episode_length

            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                total_length -= 1
                total_reward_for_episode = sum(all_rewards_in_episode)
                episode_total_rewards_list.append(total_reward_for_episode)
                all_rewards_in_episode = []

                # reset and unpack state
                state = env.reset()
                if args.natural_language:
                    (image, instruction) = state
                    instruction_indices = turn_instruction_str_to_tensor(instruction, env)
                print('Episode Over. Total Length: {}. Total reward for episode: {}. '
                      'Episode num: {}'.format(total_length,  total_reward_for_episode,
                                               len(episode_lengths)))
                print('Step no: {}. total length: {}'.format(episode_length, total_length))
                print('Rank: {}. Total Length: {}. Counter across all processes: {}. '
                      'Total reward for episode: {}'.format(rank, total_length, counter.value,
                                                            total_reward_for_episode))

            if not args.natural_language:
                image = torch.from_numpy(state)
            else:
                (image, instruction) = state
                image = torch.from_numpy(image)
                instruction_indices = turn_instruction_str_to_tensor(instruction, env)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            all_rewards_in_episode.append(reward)

            if done:
                break

        # No interaction with environment below.
        # Monitoring
        total_reward_for_num_steps = sum(rewards)
        total_reward_for_num_steps_list.append(total_reward_for_num_steps)
        avg_reward_for_num_steps = total_reward_for_num_steps / len(rewards)
        avg_reward_for_num_steps_list.append(avg_reward_for_num_steps)

        # Backprop and optimisation
        R = torch.zeros(1, 1)
        if not done:  # to change last reward to predicted value
            # todo check grayscale here
            if not args.natural_language:
                value, _, _ = model((image.unsqueeze(0).float(), (hx, cx)))
            else:
                value, _, _ = model((image.unsqueeze(0).float(),
                                     instruction_indices.long(),
                                     (tx, hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - log_probs[i] * gae.detach() - \
                          args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

        # benchmarking
        p_losses.append(policy_loss.data[0, 0])
        v_losses.append(value_loss.data[0, 0])

        if len(p_losses) > 1000:  # 1000 * 20 (args.num_steps) = 20000
            num_backprops += 1
            print(" ".join([
                "Training thread: {}".format(rank),
                "Num backprops: {}K".format(num_backprops),
                "Avg policy loss: {}".format(np.mean(p_losses)),
                "Avg value loss: {}".format(np.mean(v_losses))]))
            p_losses = []
            v_losses = []

        if rank == 0:
            print('Step no: {}. total length: {}. Time taken for args.steps ({}): {}'.format(
                episode_length,
                total_length,
                args.num_steps,
                round(time.time() - interaction_start_time, 3)))
