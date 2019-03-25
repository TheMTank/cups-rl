"""
Adapted from https://github.com/Kaixhin/Rainbow

Functions for testing Rainbow and saving graphics of statistics for rewards and Q during the
evaluation period
"""
import os
import warnings
try:
    import plotly
    from plotly.graph_objs import Scatter
    from plotly.graph_objs.scatter import Line
    plotly_installed = True
except ImportError:
    warnings.warn("Error importing plotly. No plots will be saved on evaluation")
    plotly_installed = False
import torch

from algorithms.rainbow.env import Env

""" Global variables used to track the evaluation results
eval_steps       - list of evaluation steps at each evaluation period
rewards          - list of rewards obtained at each evaluation period
Qs               - list of Q obtained at each evaluation period  
best_avg_reward  - stores the best average reward achieved to save the best model """
eval_steps, rewards, Qs, best_avg_reward = [], [], [], -1e10


# Test DQN
def test(env, num_steps, args, dqn, val_mem, evaluate_only=False):
    """
    Explanation to our multiple tests for the special case of "ai2thor":
    In AI2Thor environment the rendering is not optional, so using two instances of the environment
    is not needed, but in Atari games we need to instantiate two environments to allow the choice of
    multiple rendering options, e.g. having the option of training without rendering for efficiency
    reasons and testing with rendering.
    """
    global eval_steps, rewards, Qs, best_avg_reward
    eval_steps.append(num_steps)
    step_rewards, step_Qs = [], []
    if args.game != 'ai2thor':
        env = Env(args)
    # Test performance over several episodes
    done = True
    for episode_n in range(args.evaluation_episodes):
        print("eval episode {}/{}".format(episode_n, args.evaluation_episodes))
        step_n, reward_sum = 0, 0
        while step_n < args.max_episode_length:
            step_n += 1
            if step_n % 200 == 0:
                print("eval step {}".format(step_n))
            if done:
                state, reward_sum, done = env.reset(), 0, False
            # In evaluation we choose actions ε-greedily instead while fixing the noisy layers
            action = dqn.act_e_greedy(state)
            state, reward, done, _ = env.step(action)  # Step
            reward_sum += reward
            if args.render and args.game != 'ai2thor':
                env.render()
            if done:
                step_rewards.append(reward_sum)
    if args.game != 'ai2thor':
        env.close()
    # Test Q-values over validation memory completely independently of evaluation episodes
    for state in val_mem:  # Iterate over valid states
        step_Qs.append(dqn.evaluate_q(state))

    avg_reward, avg_Q = sum(step_rewards) / len(step_rewards), sum(step_Qs) / len(step_Qs)
    if not evaluate_only:
        # Append to results
        rewards.append(step_rewards)
        Qs.append(step_Qs)

        # Plot
        if plotly_installed:
            _plot_line(eval_steps, rewards, 'Reward', path='results')
            _plot_line(eval_steps, Qs, 'Q', path='results')

        # Save model parameters if improved
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            dqn.save(path='weights', filename='rainbow_{}.pt'.format(num_steps))

    # Return average reward and Q-value
    return avg_reward, avg_Q


def _plot_line(xs, ys_population, title, path=''):
    """ Plots min, max and mean + standard deviation bars of a population over time """
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', \
                                                       'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

    ys = torch.tensor(ys_population, dtype=torch.float32)
    ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), \
                                      ys.mean(1).squeeze(), ys.std(1).squeeze()
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(x=xs, y=ys_max.numpy(),
                        line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=ys_upper.numpy(),
                          line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour,
                         line=Line(color=mean_colour), name='Mean')
    trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour,
                          line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=ys_min.numpy(),
                        line=Line(color=max_colour, dash='dash'), name='Min')

    plotly.offline.plot({
      'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
      'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
    }, filename=os.path.join(path, title + '.html'), auto_open=False)
