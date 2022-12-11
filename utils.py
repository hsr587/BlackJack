import random
import numpy as np
import pandas as pd

import qlearn
from blackjack import Blackjack
from qlearn import QLearningAgent

ACTIONS = {0: 'Stand',
           1: 'Hit',
           2: 'Double',
           3: 'Split',
           4: 'Surrender'}

POLICY_MAP_STAND_HIT = pd.read_csv('Hit_Stand_Policy.csv', header=0, index_col=0)
DEALER_POLICY_KEY = {1: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'T'}

game = Blackjack()


def play_bj_random(game, hands=10):
    net_reward = 0
    for g in range(hands):
        player, dealer, reward, done = game.deal()
        print(str(game))
        while not done:
            action = random.choice(game.get_actions())
            player, dealer, reward, done = game.step(action)
            print(ACTIONS[action])
            print(str(game))
        net_reward += reward
        print(f'Game Finished.  Reward: {reward} Net Reward: {net_reward}\n')
    return f'{net_reward / hands * 100:.2f}%'


def play_bj_policy(game, policy, hands=10, verbose=False):
    net_reward = 0
    for g in range(hands):
        state, done, reward = game.deal()
        player, dealer = state
        if verbose:
            print(str(game))
        while not done:
            action = policy.loc[player, DEALER_POLICY_KEY[dealer]]
            state, done, reward = game.step(action)
            player, dealer = state
            if verbose:
                print(ACTIONS[action])
                print(str(game))
        net_reward += reward
        if verbose:
            print(f'Game Finished.  Reward: {reward} Net Reward: {net_reward}\n')
    return f'{net_reward / hands * 100:.2f}%'


def test_policy(game, policy, hands):
    pass


if __name__ == "__main__":
    q_agent = qlearn.QLearningAgent(game)
    q_agent.train(50000)
    new_policy = q_agent.q_table_to_policy(POLICY_MAP_STAND_HIT)
    print(new_policy)

    print(play_bj_policy(game, new_policy, 10000, verbose=False))
