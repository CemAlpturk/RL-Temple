from typing import Any

import gymnasium as gym
from gymnasium import spaces

import numpy as np

N_CARDS = 52
MAX_HAND_SIZE = 10
N_SUITS = 4
N_RANKS = 13
PLUMP = 0  # Points received for missing a bid


class PlumpEnv(gym.Env):
    """
    Environment for the card game "Plump".

    Rules:
    - Players are dealt cards (hand size decreases each round from MAX_HAND_SIZE to 1).
    - Each player bids how many tricks they expect to win.
    - Last bidder cannot bid a number that would make total bids equal to hand size.
    - Players play one card per truck, tmust follow suit if possible.
    - Highest card of led suit (or trump suit if played) wins the trick.
    - Players score points based on bids and tricks won.

    52 cards in total: 4 suits x 13 ranks
    Card index = suit * 13 + rank (0-51)
    Suits: 0=Clubs, 1=Diamonds, 3=Spades, 4=Hearts
    Ranks: 0=2, 1=3, ..., 8=10, 9=J, 10=Q, 11=K, 12=A
    """

    metadata = {"render_modes": ["human", "ansi"], "name": "Plump-v0"}

    def __init__(self, n_players: int = 2, render_mode: str | None = None) -> None:
        super().__init__()

        assert 2 <= n_players <= 5, "Number of players must be between 2 and 5."
        self.n_players = n_players

        # Action space: play a card (0-51) or bid (52 to 52 + MAX_HAND_SIZE -> 0 to MAX_HAND_SIZE)
        self.action_space = spaces.Discrete(N_CARDS + MAX_HAND_SIZE + 1)

        # Observation space
        self.observation_space = spaces.Dict(
            {
                # Current player's hand (binary encoded)
                "hand": spaces.MultiBinary(N_CARDS),
                # Current trick cards played for each player (-1 if no card played)
                "current_trick": spaces.Box(
                    low=-1,
                    high=N_CARDS - 1,
                    shape=(self.n_players,),
                    dtype=np.int8,
                ),
                # Cards already played this round (binary encoded)
                "played_cards": spaces.MultiBinary(N_CARDS),
                # Bids made by each player (-1 if no bid made)
                "bids": spaces.Box(
                    low=-1,
                    high=MAX_HAND_SIZE,
                    shape=(self.n_players,),
                    dtype=np.int8,
                ),
                # Total bids placed so far
                "total_bids": spaces.Box(
                    low=0,
                    high=MAX_HAND_SIZE * self.n_players,
                    shape=(1,),
                    dtype=np.int8,
                ),
                "blocked_bid": spaces.Box(
                    low=-1,
                    high=MAX_HAND_SIZE,
                    shape=(1,),
                    dtype=np.int8,
                ),
                # Tricks won by each player this round
                "tricks_won": spaces.Box(
                    low=0,
                    high=MAX_HAND_SIZE,
                    shape=(self.n_players,),
                    dtype=np.int8,
                ),
                # Game state
                "hand_size": spaces.Discrete(MAX_HAND_SIZE),  # Current round hand size
                "trump_suit": spaces.Discrete(5),  # 0-3 for suits, 4 for no trump
                "led_suit": spaces.Discrete(5),  # 0-3 for suits, 4 for no lead yet
                "phase": spaces.Discrete(2),  # 0: bidding, 1: playing,
                # Cumulative scores
                "scores": spaces.Box(
                    low=0,
                    high=1000,
                    shape=(self.n_players,),
                    dtype=np.int16,
                ),
                # Dealer index
                "dealer": spaces.Discrete(self.n_players),
                # Current player index
                "current_player": spaces.Discrete(self.n_players),
            }
        )

        # Internal state (initialized in reset)
        self._hands: np.ndarray  # Shape: (n_players, N_CARDS), binary encoded hands
        self._current_trick: np.ndarray
        self._played_cards: np.ndarray
        self._bids: np.ndarray
        self._tricks_won: np.ndarray
        self._scores: np.ndarray
        self._hand_size: int = 0
        self._trump_suit: int = 4
        self._led_suit: int = 4
        self._phase: int = 0
        self._dealer: int = 0
        self._current_player: int = 0
        self._trick_leader: int = 0
        self._round: int = 0
        self._deck: np.ndarray  # Shape: (N_CARDS,)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Args:
            seed (int | None): Seed for random number generator.
            options (dict[str, Any] | None): Additional options for reset.

        Returns:
            tuple[dict[str, np.ndarray], dict[str, Any]]: Initial observation and info.
        """
        super().reset(seed=seed)

        # Parse options
        start_hand_size = MAX_HAND_SIZE
        if options is not None:
            start_hand_size = options.get("start_hand_size", MAX_HAND_SIZE)

        # Initialize game state
        self._scores = np.zeros(self.n_players, dtype=np.int16)
        self._dealer = 0
        self._round = 0
        self._hand_size = start_hand_size

        # Start a new round
        self._start_new_round()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action (int): The action taken by the current player.
                - If phase == 0 (bidding): action in [51, 62] (bid 0 to 10)
                - If phase == 1 (playing): action in [0, 51] (play card)

        Returns:
            observation (dict[str, np.ndarray]): The observation after the action.
            reward (float): The reward obtained from the action.
            terminated (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated.
            info (dict[str, Any]): Additional information.
        """
        reward = 0.0
        terminated = False
        truncated = False

        # Validate action
        if not self._is_valid_action(action):
            # Invalid action - heavy penalty and skip
            # reward = -10.0
            # return self._get_observation(), reward, terminated, truncated, self._get_info()
            raise ValueError(f"Invalid action {action} in current state.")

        if self._phase == 0:  # Bidding phase
            bid = action - N_CARDS
            self._handle_bid(bid)
        else:  # Playing phase
            card = action
            round_reward = self._handle_play(card)
            if round_reward is not None:
                reward = round_reward

        # Check if game is over
        if self._hand_size == 0 and self._phase == 0:
            terminated = True

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_info(self) -> dict[str, Any]:
        """
        Get additional information about the environment.

        Returns:
            dict[str, Any]: Additional information about the environment.
        """
        return {
            "round": self._round,
            "action_mask": self.action_masks(),
        }

    def _start_new_round(self) -> None:
        """
        Initialize a new round with the current hand size.
        """

        # Reset round state
        self._hands = np.zeros((self.n_players, N_CARDS), dtype=np.int8)
        self._current_trick = -1 * np.ones(self.n_players, dtype=np.int8)
        self._played_cards = np.zeros(N_CARDS, dtype=np.int8)
        self._bids = -1 * np.ones(self.n_players, dtype=np.int8)
        self._tricks_won = np.zeros(self.n_players, dtype=np.int8)
        self._phase = 0  # Bidding phase
        self._led_suit = 4  # No lead yet

        # Shuffle and deal cards
        self._deck = self.np_random.permutation(N_CARDS)
        for i in range(self.n_players):
            start_idx = i * self._hand_size
            end_idx = start_idx + self._hand_size
            cards = self._deck[start_idx:end_idx]
            for card in cards:
                self._hands[i, card] = 1

        # Determine trump suit (last card in deck after dealing, or random)
        dealt_cards = self.n_players * self._hand_size
        if dealt_cards < N_CARDS:
            trump_card = self._deck[dealt_cards]
            self._trump_suit = trump_card // N_RANKS
        else:
            # All cards dealt, no trump or random trumo
            self._trump_suit = 4  # No trump

        # First player to bid is left of the dealer
        self._current_player = (self._dealer + 1) % self.n_players
        self._trick_leader = self._current_player

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Build the observation dictionary for the current player."""
        # Calculate blocked bid
        total_bids = np.sum(self._bids[self._bids >= 0])
        num_bid = np.sum(self._bids >= 0)

        blocked_bid = -1
        if num_bid == self.n_players - 1:
            # Current player is last to bid
            blocked_bid = self._hand_size - total_bids
            if blocked_bid < 0 or blocked_bid > self._hand_size:
                blocked_bid = -1

        return {
            "hand": self._hands[self._current_player].copy(),
            "current_trick": self._current_trick.copy(),
            "played_cards": self._played_cards.copy(),
            "bids": self._bids.copy(),
            "total_bids": np.array([total_bids], dtype=np.int8),
            "blocked_bid": np.array([blocked_bid], dtype=np.int8),
            "tricks_won": self._tricks_won.copy(),
            "hand_size": np.array(self._hand_size, dtype=np.int8),
            "trump_suit": np.array(self._trump_suit, dtype=np.int8),
            "led_suit": np.array(self._led_suit, dtype=np.int8),
            "phase": np.array(self._phase, dtype=np.int8),
            "scores": self._scores.copy(),
            "dealer": np.array(self._dealer, dtype=np.int8),
            "current_player": np.array(self._current_player, dtype=np.int8),
        }

    def _is_valid_action(self, action: int) -> bool:
        """Check if an action is valid in the current state."""
        mask = self.action_masks()
        return mask[action]

    def action_masks(self) -> np.ndarray:
        """Return a boolean mask of valid actions."""
        mask = np.zeros(N_CARDS + MAX_HAND_SIZE + 1, dtype=bool)

        if self._phase == 0:  # Bidding phase
            # Valid bids: 0 to hand_size, except blocked bid if last bidder
            total_bids = np.sum(self._bids[self._bids >= 0])
            num_bid = np.sum(self._bids >= 0)
            blocked_bid = -1

            if num_bid == self.n_players - 1:
                blocked_bid = self._hand_size - total_bids

            for bid in range(self._hand_size + 1):
                if bid == blocked_bid:
                    continue
                action_idx = N_CARDS + bid
                mask[action_idx] = True

        else:  # Playing phase
            # Valid cards: cards in hand, must follow suit of possible
            hand = self._hands[self._current_player]

            if self._led_suit == 4:
                # No card led yet, can play any card in hand
                mask[:N_CARDS] = hand.astype(bool)
            else:
                # Must follow suit if possible
                suit_start = self._led_suit * N_RANKS
                suit_end = suit_start + N_RANKS
                suit_cards = hand[suit_start:suit_end]

                if np.any(suit_cards):
                    # Have cards in led suit, must play one
                    mask[suit_start:suit_end] = suit_cards.astype(bool)
                else:
                    # Can't follow suit, play anything
                    mask[:N_CARDS] = hand.astype(bool)

        return mask

    def _handle_bid(self, bid: int) -> None:
        """Handle a player making a bid."""
        self._bids[self._current_player] = bid

        # Move to next player
        self._current_player = (self._current_player + 1) % self.n_players

        # Check if all the players have bid
        if np.all(self._bids >= 0):
            # Move to playing phase
            self._phase = 1
            self._current_player = self._trick_leader

    def _handle_play(self, card: int) -> float | None:
        """
        Handle a player playing a card.

        Returns:
            float | None: Reward for player 0 if round ended, None otherwise.
        """
        # Remove card from hand
        self._hands[self._current_player, card] = 0

        # Add to current trick
        self._current_trick[self._current_player] = card

        # Mark as played
        self._played_cards[card] = 1

        # Set led suit if first card
        if self._led_suit == 4:
            self._led_suit = card // N_RANKS

        # Move to next player
        self._current_player = (self._current_player + 1) % self.n_players

        # Check if trick is complete
        if np.all(self._current_trick >= 0):
            return self._resolve_trick()

        return None

    def _resolve_trick(self) -> float | None:
        """
        Determine winner of the trick and update state.

        Returns:
            float | None: Reward for player 0 if round ended, None otherwise.
        """
        winner = self._determine_trick_winner()
        self._tricks_won[winner] += 1

        # Reset for next trick
        self._current_trick = -1 * np.ones(self.n_players, dtype=np.int8)
        self._led_suit = 4
        self._trick_leader = winner
        self._current_player = winner

        # Check if round is complete
        if np.sum(self._tricks_won) == self._hand_size:
            return self._resolve_round()

        return None

    def _determine_trick_winner(self) -> int:
        """Determine which player won the current trick."""
        best_player = self._trick_leader
        best_card = self._current_trick[best_player]
        best_suit = best_card // N_RANKS
        best_rank = best_card % N_RANKS

        for i in range(1, self.n_players):
            player = (self._trick_leader + i) % self.n_players
            card = self._current_trick[player]
            suit = card // N_RANKS
            rank = card % N_RANKS

            # Check if this card beats the current best
            if suit == self._trump_suit and best_suit != self._trump_suit:
                # Trump beats non-trump
                best_player = player
                best_card = card
                best_suit = suit
                best_rank = rank
            elif suit == best_suit and rank > best_rank:
                # Same suit, higher rank
                best_player = player
                best_card = card
                best_suit = suit
                best_rank = rank
            elif (
                suit == self._trump_suit
                and best_suit == self._trump_suit
                and rank > best_rank
            ):
                # Both trump, higher rank
                best_player = player
                best_card = card
                best_suit = suit
                best_rank = rank

        return best_player

    def _resolve_round(self) -> float:
        """
        Calculate scores at the end of round and prepare for next round.

        Returns:
            float: Reward for player 0 (the agent) based on round outcome.
        """
        agent_reward = 0.0

        # Calculate scores - player gets a "plump" (penalty) if they don't hit their bid
        for i in range(self.n_players):
            if self._tricks_won[i] != self._bids[i]:
                # Missed bid - add a plump (penalty point)
                self._scores[i] += PLUMP
                if i == 0:  # Agent
                    agent_reward = float(PLUMP)
            else:
                # Hit bid - reward points equal to bid
                if self._bids[i] == 0:
                    points = 5  # Special case: bid 0 and succeeded
                else:
                    points = 10 + self._bids[i]  # Base 10 points + bid points
                self._scores[i] += points
                if i == 0:  # Agent
                    agent_reward = float(points)

        # Prepare for next round
        self._round += 1
        self._hand_size -= 1
        self._dealer = (self._dealer + 1) % self.n_players

        if self._hand_size > 0:
            self._start_new_round()
        else:
            # Game is ending - set phase to 0 so termination check works
            self._phase = 0

        return agent_reward

    def render(self) -> str | None:
        """Render the current game state."""
        if self.render_mode is None:
            return None

        output = []
        output.append(f"\n{'=' * 50}")
        output.append(f"Round: {self._round + 1} | Hand Size: {self._hand_size}")
        output.append(
            f"Dealer: Player {self._dealer} | Current: Player {self._current_player}"
        )
        output.append(f"Trump: {self._suit_name(self._trump_suit)}")
        output.append(f"Phase: {'Bidding' if self._phase == 0 else 'Playing'}")
        output.append(f"{'=' * 50}")

        # Show hands
        for i in range(self.n_players):
            cards = self._hand_to_string(self._hands[i])
            bid_str = str(self._bids[i]) if self._bids[i] >= 0 else "?"
            output.append(
                f"Player {i}: {cards} | Bid: {bid_str} | Tricks: {self._tricks_won[i]}"
            )

        # Show current trick
        if np.any(self._current_trick >= 0):
            trick_str = " | ".join(
                self._card_name(c) if c >= 0 else "---" for c in self._current_trick
            )
            output.append(f"\nCurrent Trick: {trick_str}")
            output.append(f"Led Suit: {self._suit_name(self._led_suit)}")

        # Show scores
        output.append(f"\nScores: {list(self._scores)}")

        result = "\n".join(output)

        if self.render_mode == "human":
            print(result)
            return None

        return result

    def _hand_to_string(self, hand: np.ndarray) -> str:
        """Convert a binary hand representation to a string."""
        cards = [self._card_name(i) for i in range(N_CARDS) if hand[i]]
        return ", ".join(cards) if cards else "(empty)"

    def _card_name(self, card_idx: int) -> str:
        """Get the name of a card from its index."""
        if card_idx < 0:
            return "---"
        suit = card_idx // N_RANKS
        rank = card_idx % N_RANKS

        suit_symbols = ["♣", "♦", "♠", "♥"]
        rank_names = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

        return f"{rank_names[rank]}{suit_symbols[suit]}"

    def _suit_name(self, suit: int) -> str:
        """Get the name of a suit."""
        if suit == 4:
            return "None"
        suit_names = ["Clubs", "Diamonds", "Spades", "Hearts"]
        return suit_names[suit]
