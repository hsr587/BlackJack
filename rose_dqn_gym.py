import random
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import numpy as np

import gym_utils

BUFFER_SIZE = 200  # replay buffer size
BATCH_SIZE = 50  # minibatch size

DISCOUNT_FACTOR = 0.99
LEARNING_RATE = .001
UPDATE_EVERY = 1
SYNC_TARGET_EPISODES = 100
EPSILON_DECAY_LAST_FRAME = 10000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
MAX_EPISODES = 50000


class DQN(nn.Module):
    """
    -------
    Neural Network Used for Agent to Approximate Q-Values
    -------
    [Params]
        'state_size' -> size of the state space
        'action_size' -> size of the action space
        'seed' -> used for random module
    """

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory():
    """
    Class to create memory for DQN training
    """

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """
        Method to allow a Transition to be added to the deque.
        :param args: arguments for adding to a Transition named tuple to be added to the list
                        should be state, action, reward, next_state, done where:
                        state: tensor, action: int, reward: float, next_state: tensor, done: bool
        :return: None
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        Method to return batch of Transitions randomly selected without replacement.
        :param batch_size: int - Length of batch
        :return:
        """
        s = random.sample(self.memory, batch_size)
        return Transition(*zip(*s))

    def __len__(self):
        return len(self.memory)


class Agent():
    """
    --------
    Deep Q-Learning Agent
    --------
    [Params]
        'state_size' -> size of the state space
        'action_size' -> size of the action space
        'seed' -> used for random module
    --------
    """

    def __init__(self, game: gym.Env, state_action_dims=[32, 11, 2, 2]):
        self.state_action_dims = state_action_dims

        # Q-Networks
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.game = game
        self.policy_net = DQN(len(state_action_dims) - 1, state_action_dims[-1]).to(device)
        self.target_net = DQN(len(state_action_dims) - 1, state_action_dims[-1]).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

        self.memory = ReplayMemory(BUFFER_SIZE)

    def get_action(self, state: torch.Tensor, epsilon=0.0):
        if random.random() < epsilon:
            return self.game.action_space.sample()
        with torch.no_grad():
            return self.policy_net(state).argmax().item()

    def learn(self, transitions):

        state_batch = torch.stack(transitions.state)  # batch size x number obs
        next_state_batch = torch.stack(transitions.next_state)  # batch size x number obs
        action_batch = torch.tensor(transitions.action).unsqueeze(1)  # batch size x 1
        reward_batch = torch.tensor(transitions.reward).unsqueeze(1)  # batch size x 1
        #print('r', reward_batch)
        done_batch = torch.tensor(transitions.done, dtype=int).unsqueeze(1)  # batch size x 1
        #print(done_batch)

        with torch.no_grad():
            target_max_qs = self.target_net(next_state_batch).max(1, keepdims=True).values  # batch_size x 1
            #print(target_max_qs)
            expected_state_action_values = reward_batch + DISCOUNT_FACTOR * target_max_qs * (1 - done_batch)  # batch_size x 1
            #print(expected_state_action_values)

        state_action_values = torch.gather(self.policy_net(state_batch), 1, action_batch)  # batch_size X 1

        loss = F.mse_loss(state_action_values.squeeze(1), expected_state_action_values.squeeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        for episode in range(1, MAX_EPISODES + 1):

            losses = []

            state = env.reset()
            state = torch.tensor((state[0] / 32, state[1] / 11, int(state[2])))
            done = False

            while not done:
                epsilon = max(EPSILON_FINAL, EPSILON_START - episode / EPSILON_DECAY_LAST_FRAME)
                action = self.get_action(state, epsilon)
                #print(epsilon, action)
                next_state, reward, done, *_ = self.game.step(action)
                next_state = torch.tensor((state[0] / 32, state[1] / 11, int(state[2])))
                #print(state, action, reward, next_state, done)
                self.memory.push(state, action, reward, next_state, done)
                state = next_state

            if episode % UPDATE_EVERY == 0 and len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                losses.append(self.learn(experiences))

            # Update target network
            if episode % SYNC_TARGET_EPISODES == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if episode % 1000 == 0:
                expected_return = self.test()
                print(
                    f'Episode {episode}  Avg loss: {sum(losses) / len(losses)}  Expected reward: {expected_return:0.4f}%')
                print(epsilon)
                #print(self.get_strategy())
                torch.save(agent.policy_net.state_dict(), 'rose_dqn_checkpoint.pth')
                losses = []

    def test(self):
        TEST_GAMES = 1000
        net_reward = 0
        for episode in range(TEST_GAMES):
            state = env.reset()
            done = False
            while not done:
                state = torch.tensor((state[0] / 32, state[1] / 11, int(state[2])))
                action = self.get_action(state)
                next_state, reward, done, *_ = env.step(action)
                net_reward += reward
                state = next_state
        return net_reward / TEST_GAMES * 100

    def get_strategy(self):
        action_map = {1: 'H', 0: 'S'}
        strategy = gym_utils.BASIC_STRATEGY_HIT_STAND_ONLY.copy(deep=True)
        self.policy_net.eval()
        for row in strategy.index:
            for col in strategy.columns:
                state = gym_utils.parse_strategy_state_to_gym_state(row, col)
                state = torch.tensor((state[0] / 32, state[1] / 11, int(state[2])))
                with torch.no_grad():
                    action = self.policy_net(state).argmax().item()
                strategy.loc[row, col] = action_map[action]
        self.policy_net.train()
        return strategy


if __name__ == '__main__':
    env = gym.make("Blackjack-v1")
    agent = Agent(env, [32, 11, 2, 2])
    agent.train()
    strategy  = agent.get_strategy()
    print(strategy)
    strategies = {'basic': gym_utils.BASIC_STRATEGY_HIT_STAND_ONLY,
                  'rose_QL': strategy,
                  'random': 'random'}
    results = gym_utils.evaluate_strategies(env, strategies)
    gym_utils.plot_results(results)

