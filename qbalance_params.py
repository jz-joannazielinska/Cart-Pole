import balance
import numpy as np

from balance_q import QLearner
from balance_sarsa import SarsaLearner

learning_rate_values = [0.1, 0.3, 0.5]
epsilon_values = [0.9, 0.85, 0.8]
buckets = [(2, 2, 8, 3), (2, 2, 6, 6), (1, 1, 6, 3)]
discount_values = [0.9, 0.98, 1.0]


def get_params():
    params = []
    for i in range(len(buckets)):
        for j in range(len(learning_rate_values)):
            for k in range(len(discount_values)):
                for e in range(len(epsilon_values)):
                    params.append([buckets[i], learning_rate_values[j], discount_values[k], epsilon_values[e]])
    return params


def perform_learning(params):
    '''
    Performs learning for each set of parameters
    :return:
    '''

    counter = 0

    for param in params:
        output_name = 'sarsa_output_' + str(counter)
        counter += 1
        learner = SarsaLearner(buckets=param[0], learning_rate=param[1], discount=param[2], epsilon=param[3],
                               output_file=output_name)
        learner.learn(10000)


def perform_multiple_learning():
    '''
    Performs learning with the same parameters multiple times to check if learning is stable
    :return:
    '''

    tries_num = 5
    best_params = [buckets[0], learning_rate_values[0], discount_values[1], epsilon_values[2]]
    for i in range(tries_num):

        output_name = 'm_output_' + str(i)
        learner1 = SarsaLearner(buckets=best_params[0], learning_rate=best_params[1], discount=best_params[2],
                                epsilon=best_params[3], output_file=output_name)
        learner1.learn(10000)
