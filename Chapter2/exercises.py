#####################################################
#                                                   #
# Exercises for Chapter 2                           #
# 2019: Nisheet Patel (nisheet.pat@gmail.com)       #
#                                                   #
#####################################################

import ten_armed_testbed
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import trange


def simulate_non_stationary(runs, time, bandits):
    rewards = np.zeros((len(bandits), int(runs), int(time)))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        print("Bandit {} of {}".format(i+1, len(bandits)))
        for r in trange(runs):
            bandit.reset()
            for t in range(time):
                bandit.q_true += np.random.randn(bandit.k)/100  # non-stationarity
                bandit.best_action = np.argmax(bandit.q_true)   # update new best action
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards


def exercise_2_5(runs=2000, time=10000):
    bandits = []
    bandits.append(ten_armed_testbed.Bandit(epsilon=0.1, sample_averages=True))
    bandits.append(ten_armed_testbed.Bandit(epsilon=0.1, step_size=0.1))
    best_action_counts, rewards = simulate_non_stationary(runs, time, bandits)

    plt.figure(figsize=(20, 10))

    plt.subplot(2, 2, 1)
    rewards_savg, = plt.plot(rewards[0])
    rewards_step, = plt.plot(rewards[1])
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend([rewards_savg, rewards_step], ['Sample averages','Constant step size'])

    plt.subplot(2, 2, 2)
    actions_savg, = plt.plot(best_action_counts[0])
    actions_step, = plt.plot(best_action_counts[1])
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend([actions_savg, actions_step], ['Sample averages','Constant step size'])

    plt.savefig('exercise_2_5.png')
    plt.close()


def exercise_2_11(runs=700, time=200000):
    labels = ['epsilon-greedy', #'gradient bandit',
              'UCB', 'optimistic initialization']
    generators = [lambda epsilon: ten_armed_testbed.Bandit(epsilon=epsilon, sample_averages=True),
                  #lambda alpha: ten_armed_testbed.Bandit(gradient=True, step_size=alpha, gradient_baseline=True),
                  lambda coef: ten_armed_testbed.Bandit(epsilon=0, UCB_param=coef, sample_averages=True),
                  lambda initial: ten_armed_testbed.Bandit(epsilon=0, initial=initial, step_size=0.1)]
    parameters = [np.arange(-7, -1, dtype=np.float),
                  #np.arange(-5, 2, dtype=np.float),
                  np.arange(-2, 7, dtype=np.float),
                  np.arange(-2, 7, dtype=np.float)]

    bandits = []
    for generator, parameter in zip(generators, parameters):
        for param in parameter:
            bandits.append(generator(pow(2, param)))

    _, average_rewards = simulate_non_stationary(runs, time, bandits)
    rewards = np.mean(average_rewards[:, int(time/2): ], axis=1)

    i = 0
    for label, parameter in zip(labels, parameters):
        l = len(parameter)
        plt.plot(parameter, rewards[i:i+l], label=label)
        i += l
    plt.xlabel('Parameter(2^x)')
    plt.ylabel('Average reward over the last {} steps'.format(int(time/2)))
    plt.legend()

    plt.savefig('exercise_2_11_{}runs.png'.format(runs))
    plt.close()


if __name__ == "__main__":
    print("Plotting exercise_2.5")
    exercise_2_5()
    #print("Plotting exercise_2.11")
    #exercise_2_11()