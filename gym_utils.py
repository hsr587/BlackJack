import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import torch

import mcguire_dqn
import rose_qlearn_gym

import qlearn
from blackjack import Blackjack
from qlearn import QLearningAgent

STRATEGY_MAP = {'S': 'S', 'H': 'H', 'Dh': 'H', 'Ds': 'S', 'Uh': 'H', 'Us': 'S'}

ALL_STRATEGIES = pd.read_csv('All_Strategies.csv', header=0, index_col=0)
BASIC_STRATEGY = ALL_STRATEGIES.drop(['S12', 'H4'])
BASIC_STRATEGY_NO_SPLIT = ALL_STRATEGIES[10:]
BASIC_STRATEGY_HIT_STAND_ONLY = BASIC_STRATEGY_NO_SPLIT.applymap(lambda s: STRATEGY_MAP[s])

DEALER_STRATEGY_MAP = {1: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'T'}
DEALER_POLICY_MAP = {v: k for k, v in DEALER_STRATEGY_MAP.items()}


def evaluate_strategies(game, strategies: dict, episodes=1000, verbose=False):
    results = {}
    for strategy in strategies.keys():
        print(f'Playing {strategy} strategy:')
        results[strategy] = []
        net_reward = 0
        for e in range(1, episodes + 1):
            reward = play_one_bj_hand(game, strategies[strategy], verbose)
            net_reward += reward
            results[strategy].append(net_reward)
            if e % 1000 == 0 and e != 0:
                print(f'{e} hands played.  Average reward per hand: {net_reward / e}')
    return results


def plot_results(results: dict):
    for k, v in results.items():
        plt.plot(range(1, len(v) + 1), v, '.-', label=k)
    plt.legend()
    plt.show()


def play_one_bj_hand(game, strategy, verbose=False):
    ACTION_MAP = {0: 'Stand', 1: 'Hit', 2: 'Double', 3: 'Surrender', 4: 'Split'}
    net_reward = 0
    state = game.reset()
    done = False
    while not done:
        action = get_action_from_strategy(state, strategy)
        new_state, reward, done, *_ = game.step(action)
        net_reward += reward
        if verbose: print(f'State: {state}  Action: {ACTION_MAP[action]}  Reward: {reward}')
        state = new_state
    if verbose: print(f'Game Finished.  Reward: {reward}')
    return net_reward


def get_action_from_strategy(state, strategy, action_space=[0, 1]):
    if type(strategy) == str and strategy == 'random':
        return random.choice(action_space)

    ACTION_MAP_TO_INT = {'S': [0], 'H': [1], 'Dh': [2, 1], 'Ds': [2, 0],
                         'Uh': [3, 1], 'Usp': [3, 1], 'Us': [3, 0], 'SP': [4]}
    row, col = parse_gym_state_to_strategy_state(state)
    actions = ACTION_MAP_TO_INT[strategy.loc[row, col]]
    return actions[0] if actions[0] in action_space else actions[1]


def parse_gym_state_to_strategy_state(state):
    p, d, s = state
    row = 'S' if s else 'H'
    row += str(p)
    col = DEALER_STRATEGY_MAP[d]
    return row, col

def parse_strategy_state_to_gym_state(row, col):
    soft = 1 if row[0]=='S' else 0
    player_value = int(row[1:])
    dealer_value = DEALER_POLICY_MAP[col]
    return player_value, dealer_value, soft

def convert_gym_qtable_to_strategy(qtable):
    action_map = {1:'H', 0:'S'}
    strategy = BASIC_STRATEGY_HIT_STAND_ONLY.copy(deep=True)
    for row in strategy.index:
        for col in strategy.columns:
            state = parse_strategy_state_to_gym_state(row, col)
            strategy.loc[row, col] = action_map[qtable[state].argmax()]
    return strategy

def convert_mcguire_dqn_to_strategy():
    action_map = {1:'H', 0:'S'}
    strategy = BASIC_STRATEGY_HIT_STAND_ONLY.copy(deep=True)
    agent = mcguire_dqn.Agent(state_size=3, action_size=2, seed=0)
    agent.qnetwork_local.load_state_dict(torch.load('mcguire_checkpoint.pth'))
    agent.qnetwork_local.eval()
    for row in strategy.index:
        for col in strategy.columns:
            state = parse_strategy_state_to_gym_state(row, col)
            state = np.array([state[0]/32, state[1]/10, int(state[2])])
            state = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                action_values = agent.qnetwork_local(state).cpu().data.numpy()
            action = action_values.argmax()
            strategy.loc[row, col] = action_map[action]
    return strategy

if __name__ == "__main__":
    env = gym.make('Blackjack-v1')
    mcguire_QL_strategy = convert_gym_qtable_to_strategy(np.load('mcguire_qtable.npy'))
    mcguire_DQN_strategy = convert_mcguire_dqn_to_strategy()

    agent = rose_qlearn_gym.QLearningAgent(env)
    agent.train(50000)
    rose_QL_strategy = convert_gym_qtable_to_strategy(agent.q_table)



    strategies = {'basic': BASIC_STRATEGY_HIT_STAND_ONLY,
                  'mcguire_QL': mcguire_QL_strategy,
                  'mcguire_DQN': mcguire_DQN_strategy,
                  'rose_QL': rose_QL_strategy,
                  'random': 'random'}
    results = evaluate_strategies(env, strategies)
    plot_results(results)
