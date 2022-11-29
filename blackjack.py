import random


class Blackjack:
    """
    Class to represent a game of BlackJack

    Attributes
    ----------
    decks : int
        the number of decks to have in the shoe for paying the game.
    allow_double : bool
        boolean to allow doubling down
    allow_split : bool or int
        if false, split not allowed, if true, max of one split allowed, if int, represents
        max splits allowed
    allow_surrender : bool
        if false, surrender not allowed, if true, surrender allowed
        # TODO early vs late surrender.
    dealer_hit_s17 : bool
        if True, dealer hits on soft 17, if false, dealer stands on soft 17

    Methods
    -------
    deal()
    step(actions)
    state()
    get_actions()
    observe()
    state()
    reset()



    """

    def __init__(self, decks=1, allow_double=False, allow_split=False,
                 allow_surrender=False, dealer_hit_s17=False):
        """

        :param decks:
        :param allow_double:
        :param allow_split:
        :param allow_surrender:
        :param dealer_hit_s17:
        """
        self.decks = decks
        self.allow_double = allow_double
        self.allow_split = allow_split
        self.allow_surrender = allow_surrender
        self.dealer_hit_s17 = dealer_hit_s17
        self.shoe = Shoe(decks)
        self.player_hand = Hand()
        self.dealer_hand = Hand()
        self.reward = 0
        self.done = False

    def deal(self):
        self.done = False
        self.reward = 0
        self.shoe = Shoe(self.decks)  # TODO multiple decks
        self.player_hand = Hand()
        self.dealer_hand = Hand()
        self.player_hand.add_card(self.shoe.deal_card())
        self.dealer_hand.add_card(self.shoe.deal_card())
        self.player_hand.add_card(self.shoe.deal_card())
        self.dealer_hand.add_card(self.shoe.deal_card())
        if self.player_hand.value == 21 and self.dealer_hand.value == 21:
            # Blackjack push
            self.reward = 0
            self.done = True
        elif self.dealer_hand.value == 21:
            # Dealer Blackjack  ##TODO hole card
            self.reward = -1
            self.done = True
        elif self.player_hand.value == 21:
            # Player Blackjack
            self.reward = +1.5
            self.done = True
        # Cards dealt, no winner
        return self.state()

    def step(self, action):
        if action not in self.get_actions():
            raise Exception("""Invalid action submitted""")
        if action == 0:  # stand
            if self.dealer_hit_s17:
                while self.dealer_hand.value <= 16 or \
                        (self.dealer_hand.soft and self.dealer_hand.value == 17):
                    self.dealer_hand.add_card(self.shoe.deal_card())
            else:
                while self.dealer_hand.value <= 16:
                    self.dealer_hand.add_card(self.shoe.deal_card())
            if self.dealer_hand.value > 21:  # Dealer busts
                self.reward = +1
            elif self.player_hand.value > self.dealer_hand.value:  # Player wins
                self.reward = +1
            elif self.player_hand.value == self.dealer_hand.value:  # Push
                self.reward = 0
            elif self.player_hand.value < self.dealer_hand.value:  # Player loses
                self.reward = -1
            self.done = True
        elif action == 1:  # hit
            self.player_hand.add_card(self.shoe.deal_card())
            if self.player_hand.value > 21:
                # Player busts
                self.reward = -1
                self.done = True
            else:
                self.reward = 0
                self.done = False
        elif action == 2:  # Double
            self.player_hand.add_card(self.shoe.deal_card())
            if self.player_hand.value > 21:
                # Player busts
                self.reward = -1
                self.done = True
            else:
                self.step(action=0)
                self.reward *= 2
        elif action == 3:  # TODO Split
            pass
        elif action == 4:  # Surrender
            self.reward = -0.5
            self.done = True

        return self.state()

    def state(self):
        if self.done:
            return self.player_hand.state, self.dealer_hand.state, self.reward, self.done
        return self.player_hand.state, self.dealer_state(), self.reward, self.done

    def dealer_state(self):
        up_card = self.dealer_hand.cards[0]
        if up_card.value == 11:
            return 1
        return up_card.value

    def get_actions(self):
        actions = None
        if not self.done:
            actions = [0, 1]
            if (self.allow_double and len(self.player_hand.cards) == 2 and
                    self.player_hand.value in [9, 10, 11]):
                actions.append(2)
            if self.allow_split and self.player_hand.state()[0] == 'P':
                actions.append(3)
            if self.allow_surrender and len(self.player_hand.cards) == 2:
                actions.append(4)
        return actions

    def observe(self):
        return self.player_hand.state, self.dealer_hand.state, self.reward, self.done

    def reset(self):
        pass

    def __str__(self):
        done = "Done" if self.done else "Ready"
        output = f'Player: {str(self.player_hand)} \n' \
                 f'Dealer: {str(self.dealer_hand)} \n' \
                 f'Reward: {self.reward} Status: {done}'
        return output


class Dealer:

    def __init__(self):
        self.hand =



class Hand:
    """
    Class to represent a blackjack hand

    Attributes
    ----------
    cards : list of Card objects
        list of cards representing each card in a hand
    value : int
        value of hand for blackjack
    soft : bool
        True if there is an ace in the hand being counted as an 11, i.e. a "soft"
        hand
    state : str
        string representation of hand for use in state space
        Examples: Hard 10 --> H10, Soft 12 --> S12, Pair of Aces --> PA

    Methods
    -------
    add_card(card)
        Method to add a card to the hand.
    update_state()
        Method to update the state of a hand after adding a card.
    """

    def __init__(self):
        self.cards = []
        self.value = 0
        self.soft = False
        self.state = None

    def add_card(self, card):
        self.cards.append(card)
        if self.soft:
            if card.rank == 'A':
                self.value += 1
            else:
                self.value += card.value
                if self.value > 21:
                    self.value -= 10
                    self.soft = False
        else:
            if card.rank == 'A':
                if self.value <= 10:
                    self.value += 11
                    self.soft = True
                else:
                    self.value += 1
            else:
                self.value += card.value
        self.state = self.update_state()

    def update_state(self):
        if len(self.cards) == 2:
            if self.value == 21:
                return "BJ"
            if self.cards[0].rank == self.cards[1].rank:
                return "P" + ("T" if self.cards[0].value == 10 else self.cards[0].rank)
        return ('S' if self.soft else 'H') + str(self.value)

    def __str__(self):
        return ' '.join(str(_) for _ in self.cards) + '  ' + self.state


class Shoe:
    """
    Class to represent cards to be dealt from in Blackjack.

    Attributes
    ----------
    num_decks : int
        if zero, cards are randomly chosen from an infinite deck, otherwise this is an
        int to represent number of decks in the shoe.

    Methods
    -------
    deal_card()
        Returns card from the top of the deck.
    """

    def __init__(self, num_decks=1):
        """
        Constructor for shoe.

        :param num_decks: int
            if zero, cards are randomly chosen from an infinite deck, otherwise this is an
            int to represent number of decks in the shoe.
        """
        self.num_decks = num_decks
        deck = Deck()
        self.cards = deck
        if num_decks:
            self.cards = deck.cards * num_decks
            random.shuffle(self.cards)

    def deal_card(self):
        """
        Method to return top card from shoe.  If num decks = 0 (infinite shoe), cad return is
        random card from unshuffled deck, else, cad is top cad of shuffled shoe.

        :return: Returns Card object from top of Deck.
        """
        if self.num_decks:
            return self.cards.pop()
        return random.choice(self.cards)


class Deck:
    """
    Class to represent a deck of 52 cards

    Attributes
    ----------
    None

    Methods
    -------
    None
    """

    def __init__(self):
        """
        Constructor for deck of 52 cards.

        """
        self.cards = []
        for r in range(13):
            for s in range(4):
                self.cards.append(Card(r + 1, s))


class Card:
    """
    Class to represent individual cards in a deck.

    Attributes
    ----------
    int_rank : int
        int from 1 to 13 representing each rank of card
    int_suit : int
        int from 0 to 3 representing each possible suit of card
    rank : str
        char for rank of each card from A23456789TJQK
    suit : str
        char for suit of each card from ♣♦♥♠
    value : int
        value of card for blackjack.  Ace has value of 11.
    face : str
        string representing rank and suit of card.

    Methods
    -------
    None
    """

    def __init__(self, int_rank, int_suit):
        """

        :param int_rank:
        :param int_suit:
        """
        ranks = {1: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                 8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K'}
        suits = {0: '♣', 1: '♦', 2: '♥', 3: '♠'}
        self.int_rank = int_rank
        self.int_suit = int_suit
        self.rank = ranks[int_rank]
        self.suit = suits[int_suit]
        self.value = int_rank if int_rank < 10 else 10
        self.value = 11 if int_rank == 1 else self.value
        self.face = self.rank + self.suit

    def __add__(self, other_card):
        return self.value + other_card.value

    def __str__(self):
        return self.face
