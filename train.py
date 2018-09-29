"""
Adapted from https://github.com/ikostrikov/pytorch-a3c
"""



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

MOVEMENT_REWARD_DISCOUNT = -0.001

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock, optimizer=None):
    torch.manual_seed(args.seed + rank)

    # env = create_atari_env(args.env_name)
    env = envs.ThorWrapperEnv()
    env.seed(args.seed + rank)

    # model = ActorCritic(env.observation_space.shape[0], env.action_space)
    model = ActorCritic(1, env.action_space)
    # model = ActorCritic(3, env.action_space)

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
    # plt.ion()
    # plt.ioff()  # turn of interactive plotting mode

    total_length = 0
    episode_length = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
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
            # state, reward, done, _ = env.step(action.numpy()[0][0])
            state, reward, done = env.step(action_int)

            reward += MOVEMENT_REWARD_DISCOUNT

            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                total_length -= 1
                state = env.reset()
                every_x_training_steps = 50
                num_elements_avg = 5
                # if total_length % int(args.num_steps * every_x_training_steps) == 0:
                # print('Step no: {}. total length: {}. '.format(episode_length, total_length,
                #                                                                  avg_reward_for_num_steps))
                print('total_length % (args.num_steps * every_x_training_steps) == 0')
                print('{} % ({} * {}) == 0'.format(total_length, args.num_steps, every_x_training_steps))
                # x = [i for i in range(total_length) if i % args.num_steps == 0][-50:]
                # avg_avg_rewards = sum([x for x in avg_reward_for_num_steps_list if i ])

                mean = lambda x: sum(x) / len(x)
                avg_avg_rewards = [mean(avg_reward_for_num_steps_list[i:i + num_elements_avg]) for i in
                                   range(0, len(avg_reward_for_num_steps_list), num_elements_avg)]
                total_reward_averages = [mean(total_reward_for_num_steps_list[i:i + num_elements_avg]) for i in
                                         range(0, len(total_reward_for_num_steps_list), num_elements_avg)]
                # todo get averages of periods of 100 [0:5], [5, 10]
                x = range(len(avg_avg_rewards))

                assert len(total_reward_averages) == len(avg_avg_rewards)
                fig = plt.figure(2)
                plt.clf()
                # fig1 = plt.gcf()

                try:
                    plt.plot(x, avg_avg_rewards)
                    plt.plot(x, total_reward_averages)
                except Exception as e:
                    import pdb; pdb.set_trace()
                plt.pause(0.001)

                fp = '/home/beduffy/all_projects/ai2thor-testing/pictures/a3c-total-step-{}.png'.format(
                    total_length)
                plt.savefig(fp)
                print('saved avg acc to: {}'.format(fp))
                plt.close(fig)

                # next figure
                total_reward_for_episode = sum(all_rewards_in_episode)
                episode_total_rewards_list.append(total_reward_for_episode)

                fig = plt.figure(1)
                plt.clf()
                x = range(len(episode_total_rewards_list))
                y = episode_total_rewards_list
                plt.plot(x, y)
                fp = '/home/beduffy/all_projects/ai2thor-testing/pictures/a3c-total-reward-per-episode-{}.png'.format(
                    total_length)
                plt.savefig(fp)
                print('saved avg acc to: {}'.format(fp))
                plt.close(fig)
                # plt.show()
                # plt.draw()

                print('Total Length: {}. done after reset: {}. Total reward for episode: {}'.format(total_length, done, total_reward_for_episode))

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
