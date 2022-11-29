import random

from blackjack import Blackjack
import pandas as pd

ACTIONS = {0: 'Stand',
           1: 'Hit',
           2: 'Double',
           3: 'Split',
           4: 'Surrender'}

POLICY_MAP_STAND_HIT = pd.read_csv('Hit_Stand_Policy.csv', header=0, index_col=0)

game = Blackjack()


# game.deal()
# print(str(game))
#
#
# while not game.observe()[3]:
#     game.step(0)
#     print('stand')
#     print(str(game))

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
        player, dealer, reward, done = game.deal()
        if verbose:
            print(str(game))
        while not done:
            action = policy.loc[player, dealer]
            player, dealer, reward, done = game.step(action)
            if verbose:
                print(ACTIONS[action])
                print(str(game))
        net_reward += reward
        if verbose:
            print(f'Game Finished.  Reward: {reward} Net Reward: {net_reward}\n')
    return f'{net_reward / hands * 100:.2f}%'


print(play_bj_policy(game, POLICY_MAP_STAND_HIT, 10, verbose=True))



def q_learning(game):
    pass

