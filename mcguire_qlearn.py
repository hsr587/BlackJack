import gym
import numpy as np
import random
import matplotlib.pyplot as plt

if gym.__version__ > "0.25.0":
    print("must use gym 0.25")
    exit()


def get_state_idxs(state):
    idx1, idx2, idx3 = state
    idx3 = int(idx3)
    return idx1, idx2, idx3


def update_qtable(qtable, state, action, reward, next_state, alpha, gamma):
    curr_idx1, curr_idx2, curr_idx3 = get_state_idxs(state)
    next_idx1, next_idx2, next_idx3 = get_state_idxs(next_state)
    curr_state_q = qtable[curr_idx1][curr_idx2][curr_idx3]
    next_state_q = qtable[next_idx1][next_idx2][next_idx3]
    qtable[curr_idx1][curr_idx2][curr_idx3][action] += alpha * (
                reward + gamma * np.max(next_state_q) - curr_state_q[action])
    return qtable


def get_action(qtable, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        idx1, idx2, idx3 = get_state_idxs(state)
        action = np.argmax(qtable[idx1][idx2][idx3])
    return action


def train_agent(env,
                qtable: np.ndarray,
                num_episodes: int,
                alpha: float,
                gamma: float,
                epsilon: float,
                epsilon_decay: float) -> np.ndarray:
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while True:
            action = get_action(qtable, state, epsilon)
            new_state, reward, done, info = env.step(action)
            qtable = update_qtable(qtable, state, action, reward, new_state, alpha, gamma)
            state = new_state
            if done:
                break
        epsilon = np.exp(-epsilon_decay * episode)
    return qtable


def watch_trained_agent(env, qtable, num_rounds):
    #envdisplay = JupyterDisplay(figsize=(10, 6))
    rewards = []
    for s in range(1, num_rounds + 1):
        state = env.reset()
        done = False
        round_rewards = 0
        while True:
            action = get_action(qtable, state, epsilon)
            new_state, reward, done, info = env.step(action)
            envdisplay.show(env)

            round_rewards += reward
            state = new_state
            if done == True:
                break
        rewards.append(round_rewards)
    return rewards


env = gym.make("Blackjack-v1")
env.action_space.seed(42)

# get initial state
state = env.reset()

state_size = [x.n for x in env.observation_space]
action_size = env.action_space.n

qtable = np.zeros(state_size + [action_size])  # init with zeros

alpha = 0.3  # learning rate
gamma = 0.1  # discount rate
epsilon = 0.9  # probability that our agent will explore
decay_rate = 0.005

# training variables
num_hands = 500_000

qtable = train_agent(env,
                     qtable,
                     num_hands,
                     alpha,
                     gamma,
                     epsilon,
                     decay_rate)

np.save('mcguire_qtable.npy', qtable)


# print(f"Qtable Max: {np.max(qtable)}")
# print(f"Qtable Mean: {np.mean(qtable)}")
# print(f"Qtable Num Unique Vals: {len(np.unique(qtable))}")
#
#
# # Watch trained agent
# env = gym.make("Blackjack-v1")
# #env.action_space.seed(42)
# rewards = watch_trained_agent(env, qtable, num_rounds=100)
# env.close()
#
# plt.figure(figsize=(12,8))
# plt.plot(np.cumsum(rewards))
# plt.ylabel('Score')
# plt.xlabel('Episode')
# plt.title("Total Rewards Over Time")
# plt.show()
