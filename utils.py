import time
import shutil

import torch
import numpy as np
import matplotlib as mpl
# mpl.use('TkAgg')  # or whatever other backend that you want
mpl.use('Agg')  # or whatever other backend that you want
import matplotlib.pyplot as plt

def create_plots(experiment_id, avg_reward_for_num_steps_list, total_reward_for_num_steps_list, number_of_episodes,
                 episode_total_rewards_list, episode_lengths, env, prob, p_losses, v_losses, print_all):

    # Plotting of ery bad chart of average of rewards
    num_elements_avg = 5
    mean = lambda x: sum(x) / len(x)
    avg_avg_rewards = [mean(avg_reward_for_num_steps_list[i:i + num_elements_avg]) for i in
                       range(0, len(avg_reward_for_num_steps_list), num_elements_avg)]
    total_reward_averages = [mean(total_reward_for_num_steps_list[i:i + num_elements_avg]) for i in
                             range(0, len(total_reward_for_num_steps_list), num_elements_avg)]
    # todo get averages of periods of 100 [0:5], [5, 10]
    x = range(len(avg_avg_rewards))

    assert len(total_reward_averages) == len(avg_avg_rewards)
    fig = plt.figure(1)
    plt.clf()
    # fig1 = plt.gcf()
    plt.plot(x, avg_avg_rewards, label='label avg of avg rewards')
    plt.plot(x, total_reward_averages, label='Avg of total rewards within 20 steps')
    plt.title('Avg avg rewards averaged over 20 steps and then over 5 20 steps. Same done for total')
    plt.legend()
    plt.pause(0.001)

    # todo relative paths don't work, why?
    fp = '/home/beduffy/all_projects/ai2thor-testing/experiments/{}/a3c-avgavg-reward-and-avg-total-reward-num-episodes-{}.png'.format(experiment_id,
          number_of_episodes)
    plt.savefig(fp)
    print('saved avg acc to: {}'.format(fp))
    plt.close(fig)

    # Plot otal Reward per episode chart
    fig = plt.figure(2)
    plt.clf()
    x = range(len(episode_total_rewards_list))
    y = episode_total_rewards_list
    plt.plot(x, y)
    plt.title('Total Reward per episode Episode num: {}'.format(number_of_episodes))
    fp = '/home/beduffy/all_projects/ai2thor-testing/experiments/{}/a3c-total-reward-per-episode-{}.png'.format(
         experiment_id, number_of_episodes)
    plt.savefig(fp)
    print('saved total reward per episode to: {}'.format(fp))
    plt.close(fig)

    # Plot episode lengths chart
    fig = plt.figure(3)
    plt.clf()
    x = range(len(episode_lengths))
    y = episode_lengths
    plt.plot(x, y)
    plt.title('Episode lengths per episode Episode num: {}'.format(number_of_episodes))
    fp = '/home/beduffy/all_projects/ai2thor-testing/experiments/{}/episode-lengths-{}.png'.format(experiment_id,
         number_of_episodes)
    plt.savefig(fp)
    print('Saved episode lengths to: {}'.format(fp))
    plt.close(fig)
    # todo fill bottom underneath line so less coloured in is better

    # Plot action probabilities chart
    x_tick_labels = [env.ACTION_SPACE[i]['action'] for i in range(len(env.ACTION_SPACE))]
    x_pos = np.arange(len(x_tick_labels))
    probabilities = prob.data.numpy()[0]

    # todo something went wrong with the x labels being floats and 0.5

    fig = plt.figure(4)
    plt.clf()
    plt.bar(x_pos, probabilities, align='center', alpha=0.5)
    plt.xticks(x_pos, x_tick_labels, rotation='vertical')
    plt.ylabel('Probability')
    plt.title('Action probabilities Episode num: {}'.format(number_of_episodes))
    fp = '/home/beduffy/all_projects/ai2thor-testing/experiments/{}/action-probabilities-{}.png'.format(experiment_id,
                                                                                                   number_of_episodes)
    plt.savefig(fp)
    print('Saved action probabilities to: {}'.format(fp))
    plt.close(fig)

    # print(x_pos)
    # print(x_tick_labels)
    # print(probabilities)

    # plot policy loss charts
    x1 = range(len(p_losses))
    y1 = p_losses

    fig = plt.figure(5)
    plt.clf()
    plt.plot(x1, y1)
    plt.ylabel('Policy loss')
    plt.title('Policy loss. Episode num: {}'.format(number_of_episodes))
    fp = '/home/beduffy/all_projects/ai2thor-testing/experiments/{}/policy-loss-{}.png'.format(experiment_id,
                                                                                                        number_of_episodes)
    plt.savefig(fp)
    print('Saved Policy loss to: {}'.format(fp))
    plt.close(fig)

    # plot value loss chart
    x2 = range(len(v_losses))
    y2 = v_losses

    fig = plt.figure(6)
    plt.clf()
    plt.plot(x2, y2)
    plt.ylabel('Value loss')
    plt.title('Value loss. Episode num: {}'.format(number_of_episodes))
    fp = '/home/beduffy/all_projects/ai2thor-testing/experiments/{}/value-loss-{}.png'.format(experiment_id,
                                                                                                        number_of_episodes)
    plt.savefig(fp)
    print('Saved Value loss to: {}'.format(fp))
    plt.close(fig)


def save_checkpoint(state, experiment_id, filename, is_best=False):
    fp = '/home/beduffy/all_projects/ai2thor-testing/experiments/{}/{}'.format(experiment_id, filename)
    torch.save(state, fp)
    print('Saved model to path: {}'.format(fp))
    # if is_best:
    #     shutil.copyfile(filepath, 'model_best.pth.tar')

# todo create more plots of gradients at different layers etc
