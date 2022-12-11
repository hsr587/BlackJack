import random

import numpy as np
import pandas as pd
import gym


class QLearningAgent:

    def __init__(self, game, s_a_dims=[32, 11, 2, 2]):
        self.q_table = np.zeros(s_a_dims)
        print(self.q_table.shape)
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.game = game
        self.epsilon = 1.0
        self.epsilon_decay = 0.000005

    def train(self, num_episodes):

        prior_q_table = self.q_table.copy()
        for episode in range(num_episodes):
            state = self.game.reset()
            state = (state[0], state[1], int(state[2]))
            done = False
            while not done:
                action = self.get_action(state)
                new_state, reward, done, *_ = self.game.step(action)
                new_state = (state[0], state[1], int(state[2]))
                self.update_q_table(state, action, reward, new_state, done)
                state = new_state
            self.epsilon = np.exp(-self.epsilon_decay * episode)
            if episode % 10000 == 0 and episode > 0:
                print(self.epsilon)
                max_change = np.max(np.abs(self.q_table - prior_q_table))
                if max_change < 0.1:
                    print(f'Converged at {episode} episodes. Max delta: {max_change}')
                    break
                else:
                    print(f'Updating Q Table, episode {episode}.  Max change: {max_change:0.4f} Qmax:{self.q_table.max():0.4f} Qmin: {self.q_table.min():0.4f}...')
                    prior_q_table = self.q_table.copy()
        print(f'Q Table Updates Complete.')

    def get_action(self, state):
        action_space = [0, 1]
        if random.random() < self.epsilon:
            action = random.choice(action_space)
        else:
            action = self.q_table[state].argmax()
        return action

    def update_q_table(self, state, action, reward, new_state, done):
        q_table = self.q_table
        lr = self.learning_rate
        df = self.discount_factor
        sa = state + (action,)
        q_table[sa] += lr * (reward + df * q_table[new_state].max() - q_table[sa])
        # max_exp_future_reward = 0 if done else np.max(q_table[new_state])
        # q_table[sa] += lr * (reward + df * max_exp_future_reward - q_table[sa])

    def converged(self, new_q_table, prior_q_table, max_diff=0.01):
        diff = np.abs(new_q_table - prior_q_table)
        return np.max(diff) < max_diff
