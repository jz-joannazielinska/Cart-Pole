import math
import numpy as np
import random
import gym
from collections import defaultdict
from operator import itemgetter
from matplotlib import pyplot as plt


class SarsaLearner:
    def __init__(self, buckets=(1, 1, 6, 3),  min_lr=0.1, min_epsilon=0.1, discount=1.0, decay=25,
                 learning_rate=0.0, epsilon=0.0, output_file=None):
        self.environment = gym.make('CartPole-v1')
        self.attempt_no = 1
        self.upper_bounds = [
            self.environment.observation_space.high[0],
            0.5,
            self.environment.observation_space.high[2],
            math.radians(50)
        ]
        self.lower_bounds = [
            self.environment.observation_space.low[0],
            -0.5,
            self.environment.observation_space.low[2],
            -math.radians(50)
        ]
        self.Q_dict = defaultdict(lambda: 0)
        self.buckets=buckets

        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.ep_rewards = []
        self.ep_avg = []
        self.ep_length = []
        self.output_file = open(output_file, "w+")

    def learn(self, max_attempts):
        self.output_file.write("Buckets, Learning rate, Discount, Epsilon\n")
        self.output_file.write(str(self.buckets) + "," + str(self.learning_rate) + ", " + str(self.discount) + ", " + str(self.epsilon) +"\n")
        start = 0
        for i in range(max_attempts):
            reward_sum = self.attempt(i)
            print("Episode {} reward sum: {}".format(i, reward_sum))
            self.output_file.write(str(i) + ", " + str(reward_sum)+"\n")
            self.ep_rewards.append(reward_sum)
            self.ep_avg.append(np.mean(self.ep_rewards[min(start, i-100):]))
            start = i

    def attempt(self, i):
        observation = self.discretise(self.environment.reset())
        done = False
        reward_sum = 0.0

        if i > 2000 :  self.epsilon *= 0.997**i
        steps = 0

        while not done:
            # self.environment.render()
            action = self.pick_action(observation)
            new_observation, reward, done, info = self.environment.step(action)
            new_observation = self.discretise(new_observation)
            new_action = self.pick_action(new_observation)
            self.update_knowledge(action, observation, new_observation, new_action, reward)
            observation = new_observation
            reward_sum += reward
            steps = steps+1

        self.attempt_no += steps
        self.ep_length.append(steps)
        return reward_sum

    def discretise(self, obs):

        ratios = [(obs[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]

        return tuple(new_obs)

    def pick_action(self, observation):
        if random.uniform(0, 1) < self.epsilon:
            return self.environment.action_space.sample()
        else:
            vals = dict((k, self.Q_dict[k]) for k in self.Q_dict.keys() if k[0] == observation)
            if not vals:
                return self.environment.action_space.sample()
            else:
                return max(vals.items(), key=itemgetter(1))[0][1]

    def update_knowledge(self, action, observation, new_observation, new_action, reward):
        old_value = self.Q_dict[(observation, action)]
        new_value = self.Q_dict[(new_observation, new_action)]
        new_value = self.learning_rate * (reward + self.discount * new_value - old_value)
        self.Q_dict[(observation, action)] += new_value

    def get_next_max(self, new_observation):
        vals = dict((k, self.Q_dict[k]) for k in self.Q_dict.keys() if k[0] == new_observation)
        if not vals:
            return 0
        return max(vals.items(), key=itemgetter(1))[1]
