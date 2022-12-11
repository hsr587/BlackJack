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
        self.player = Player()
        self.dealer = Dealer()
        self.reward = 0
        self.done = False

    def deal(self):
        self.done = False
        self.reward = 0
        self.shoe = Shoe(self.decks)  # TODO multiple decks
        cards = [self.shoe.deal_card() for _ in range(4)]
        self.player = Player(cards[0], cards[2])
        self.dealer = Dealer(cards[1], cards[3])

        if self.player.state == 'BJ' and self.dealer.value == 21:  # Blackjack push
            self.reward = 0
            self.done = True
        elif self.dealer.value == 21:  # Dealer Blackjack
            self.reward = -1
            self.done = True
        elif self.player.state == 'BJ':  # Player Blackjack
            self.reward = +1.5
            self.done = True
        else:  # Cards dealt, no winner
            pass

        return self.state, self.done, self.reward

    def step(self, action):
        if action not in self.get_actions():
            raise Exception("""Invalid action submitted""")

        if action == 0:  # stand
            self.dealer.play(self.shoe, self.dealer_hit_s17)
            if self.dealer.value > 21:  # Dealer busts
                self.reward = +1
            elif self.player.hand.value > self.dealer.value:  # Player hand wins
                self.reward = +1
            elif self.player.hand.value < self.dealer.value:  # Player hand loses
                self.reward = -1
            else:  # Push
                self.reward = 0
            self.done = True

        elif action == 1:  # hit
            current_hand = self.player.hand
            current_hand.add_card(self.shoe.deal_card())
            if current_hand.value > 21:  # Player busts current hand
                self.reward = -1
                self.done = True
            else:
                self.reward = 0
                self.done = False

        elif action == 2:  # Double
            current_hand = self.player.hand
            current_hand.add_card(self.shoe.deal_card())
            if current_hand.value > 21:  # Player busts
                self.reward = -2
                self.done = True
            else:
                self.dealer.play(self.dealer_hit_s17)
                if self.dealer.value > 21:  # Dealer busts
                    self.reward = +2
                elif self.player.hand.value > self.dealer.value:  # Player hand wins
                    self.reward = +2
                elif self.player.hand.value < self.dealer.value:  # Player hand loses
                    self.reward = -2
                else:  # Push
                    self.reward = 0
                self.done = True

        elif action == 3:  # No Split in this version
            pass

        elif action == 4:  # Surrender
            self.reward = -0.5
            self.done = True

        return self.state, self.done, self.reward

    @property
    def state(self):
        return self.player.state, self.dealer.state

    def get_actions(self):
        actions = None
        if not self.done:
            actions = [0, 1]
            if self.allow_double and len(self.player.hand.cards) == 2 and self.player.hand.value in [9, 10, 11]:
                actions.append(2)
            if self.allow_split and self.player.hand.state[0] == 'P':
                actions.append(3)
            if self.allow_surrender and len(self.player.hand.cards) == 2:
                actions.append(4)
        return actions

    def __str__(self):
        done = "Done" if self.done else "Ready"
        output = f'Player: {str(self.player.hand)} \n' \
                 f'Dealer: {str(self.dealer)} \n' \
                 f'Reward: {self.reward} Status: {done}'
        return output


class Player:

    def __init__(self, card_1=None, card_2=None):
        self.hand = Hand(card_1, card_2)

    def add_card(self, card):
        self.hand.add_card(card)

    def __str__(self):
        pass

    @property
    def state(self):
        return self.hand.state


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

    def __init__(self, card_1=None, card_2=None):
        if card_1 and card_2:
            self.cards = [card_1, card_2]
        elif card_1:
            self.cards = [card_1]
        else:
            self.cards = []

        self.soft = None
        self.value = None
        self.count()
        self.bet = 1

    def add_card(self, card):
        self.cards.append(card)
        self.count()

    def count(self):
        values = [c.value for c in self.cards]
        value = sum(values)
        if 1 not in values or value + 10 > 21:
            self.value = value
            self.soft = False
        else:
            self.value = value + 10
            self.soft = True

    @property
    def state(self):
        if len(self.cards) == 2:
            if self.value == 21:
                return "BJ"
            if self.cards[0].rank == self.cards[1].rank:
                return "P" + ("T" if self.cards[0].value == 10 else self.cards[0].rank)
        return ('S' if self.soft else 'H') + str(self.value)

    def __str__(self):
        return ' '.join(str(c) for c in self.cards) + '  ' + self.state


class Dealer(Hand):

    def __init__(self, card_1=None, card_2=None):
        super().__init__(card_1, card_2)

    @property
    def state(self):
        return self.cards[0].value

    def play(self, shoe, dealer_hit_s17=False):
        if dealer_hit_s17:
            while self.value <= 16 or (self.soft and self.value == 17):
                self.add_card(shoe.deal_card())
        else:
            while self.value <= 16:
                self.add_card(shoe.deal_card())

    def __str__(self):
        return ' '.join(str(c) for c in self.cards) + '  Total: ' + str(self.value)


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
        value of card for blackjack.  Ace has value of 1.

    Methods
    -------
    add
    str
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

    def __add__(self, other_card):
        return self.value + other_card.value

    def __str__(self):
        return self.rank + self.suit
