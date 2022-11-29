import random
import pandas as pd

class Q_Learning_Agent:





    def __init__(self):
        self.q_table = np.zeros([2,2,32,5])
        self.learning_rate =
        self.discount =
        self.game = game
        self.explore_probability =

    def train(self, game):

        player, dealer, reward, done = game.deal()
        current_state = player, dealer

        while not converged():
            if explore():
                action = random.choice(game.get_actions())
            else:
                action = policy.loc[current_state]  # TODO needs to be fixed for q tabel

            next_player, next_dealer, next_reward, next_done = game.step(action)

            qtable[player, dealer, action] = (1-self.learning_rate)

    def to_np_state(self, str_state):
        player_str, dealer_str = str_state
        pair = False
        soft = False
        player_value = player_str[1:]
        dealer_value = dealer
        player_str, dealer_str = str_state
        if player_str[0] == 'P':
            pair = True
        elif player_str[0] == 'S':
            soft = True











    def explore(self):
        return random.random() < self.explore_probability