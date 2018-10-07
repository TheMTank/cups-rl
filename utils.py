import time

import numpy as np
import matplotlib as mpl
# mpl.use('TkAgg')  # or whatever other backend that you want
mpl.use('Agg')  # or whatever other backend that you want
import matplotlib.pyplot as plt

def create_plots(experiment_id, avg_reward_for_num_steps_list, total_reward_for_num_steps_list, number_of_episodes,
                 episode_total_rewards_list, episode_lengths):
    num_elements_avg = 5
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
    plt.plot(x, avg_avg_rewards, label='label avg of avg rewards')
    plt.plot(x, total_reward_averages, label='Avg of total rewards within 20 steps')
    plt.title('Avg avg rewards averaged over 20 steps and then over 5 20 steps. Same done for total')
    plt.legend()
    plt.pause(0.001)

    # todo relative paths don't work, why?
    fp = '/home/beduffy/all_projects/ai2thor-testing/experiments/{}/a3c-num-episodes-{}.png'.format(experiment_id,
          number_of_episodes)
    plt.savefig(fp)
    print('saved avg acc to: {}'.format(fp))
    plt.close(fig)

    # next figure
    fig = plt.figure(1)
    plt.clf()
    x = range(len(episode_total_rewards_list))
    y = episode_total_rewards_list
    plt.plot(x, y, label='Total Reward per episode')
    plt.title('Total Reward per episode')
    plt.legend()
    fp = '/home/beduffy/all_projects/ai2thor-testing/experiments/{}/a3c-total-reward-per-episode-{}.png'.format(
         experiment_id, number_of_episodes)
    plt.savefig(fp)
    print('saved avg acc to: {}'.format(fp))
    plt.close(fig)

    fig = plt.figure(3)
    plt.clf()
    x = range(len(episode_lengths))
    y = episode_lengths
    plt.plot(x, y)
    plt.title('Episode lengths per episode')
    fp = '/home/beduffy/all_projects/ai2thor-testing/experiments/{}/episode-lengths-{}.png'.format(experiment_id,
         number_of_episodes)
    plt.savefig(fp)
    print('saved avg acc to: {}'.format(fp))
    plt.close(fig)
    # plt.show() # for live mode but doesn't work
    # plt.draw()
