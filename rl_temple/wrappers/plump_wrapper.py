"""
Wrapper for the Plump environment to make it compatible with standard RL training.

Features:
- Flattens Dict observation space to Box space
- Auto-plays opponent turns with random or custom policy
- Exposes action_masks() for masked action sampling
- Normalizes observations to [0, 1] range
"""

from typing import Any, Callable

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from rl_temple.environments.plump import PlumpEnv, N_CARDS, MAX_HAND_SIZE


class PlumpWrapper(gym.Wrapper):
    """
    Wraps PlumpEnv for single-agent RL training.

    - Flattens Dict observations to a single vector
    - Automatically plays opponent turns
    - Provides action masks for valid action sampling
    """

    def __init__(
        self,
        env: PlumpEnv,
        opponent_policy: Callable[[np.ndarray, np.ndarray], int] | None = None,
        normalize_obs: bool = True,
    ) -> None:
        """
        Args:
            env: The PlumpEnv instance
            opponent_policy: Optional callable(obs, action_mask) -> action for opponents.
                           If None, uses random valid actions.
            normalize_obs: Whether to normalize observations to [0, 1] range
        """
        super().__init__(env)

        self.opponent_policy = opponent_policy
        self.normalize_obs = normalize_obs
        self.agent_player = 0  # The RL agent is always player 0

        # Calculate flattened observation size
        # hand: 52, current_trick: n_players, played_cards: 52,
        # bids: n_players, total_bids: 1, blocked_bid: 1,
        # tricks_won: n_players, hand_size: 1, trump_suit: 1,
        # led_suit: 1, phase: 1, scores: n_players, dealer: 1, current_player: 1
        n_players = env.n_players
        self.obs_size = (
            N_CARDS  # hand
            + n_players  # current_trick
            + N_CARDS  # played_cards
            + n_players  # bids
            + 1  # total_bids
            + 1  # blocked_bid
            + n_players  # tricks_won
            + 1  # hand_size
            + 1  # trump_suit
            + 1  # led_suit
            + 1  # phase
            + n_players  # scores
            + 1  # dealer
            + 1  # current_player
        )

        # Override observation space
        self.observation_space = spaces.Box(
            low=0.0 if normalize_obs else -1.0,
            high=1.0 if normalize_obs else 1000.0,
            shape=(self.obs_size,),
            dtype=np.float32,
        )

        # Action space remains the same
        self.action_space = env.action_space

        # Store last action mask for external access
        self._action_mask: np.ndarray = np.zeros(N_CARDS + MAX_HAND_SIZE + 1, dtype=bool)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset and play opponent turns until it's the agent's turn."""
        obs_dict, info = self.env.reset(seed=seed, options=options)
        plump_env: PlumpEnv = self.env  # type: ignore

        # Play opponent turns if needed
        obs_dict, info, _, _ = self._play_opponents(obs_dict, info)

        # Refresh action mask for the agent (current player should be agent now)
        self._action_mask = plump_env.action_masks()
        info["action_mask"] = self._action_mask

        # Flatten observation
        obs = self._flatten_obs(obs_dict)

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take agent action, then play opponent turns."""
        plump_env: PlumpEnv = self.env  # type: ignore

        # Agent takes action
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        total_reward = float(reward)

        if terminated or truncated:
            obs = self._flatten_obs(obs_dict)
            self._action_mask = info["action_mask"]
            return obs, total_reward, terminated, truncated, info

        # Play opponent turns
        obs_dict, info, opponent_reward, opp_terminated = self._play_opponents(
            obs_dict, info
        )

        # Accumulate reward from opponent phase (agent's round rewards)
        total_reward += opponent_reward

        # If opponents finished the game
        if opp_terminated:
            terminated = True

        # Also check termination after opponent turns
        if plump_env._hand_size == 0 and plump_env._phase == 0:
            terminated = True

        obs = self._flatten_obs(obs_dict)

        # Refresh action mask for the agent (current player should be agent now)
        if not terminated:
            self._action_mask = plump_env.action_masks()
            info["action_mask"] = self._action_mask
        else:
            self._action_mask = info["action_mask"]

        return obs, total_reward, terminated, truncated, info

    def _play_opponents(
        self,
        obs_dict: dict[str, np.ndarray],
        info: dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any], float, bool]:
        """
        Play opponent turns until it's the agent's turn or game ends.

        Returns:
            Updated observation, info, accumulated reward, and terminated flag
        """
        total_reward = 0.0
        terminated = False
        plump_env: PlumpEnv = self.env  # type: ignore

        while obs_dict["current_player"] != self.agent_player:
            # Check if game is over
            if plump_env._hand_size == 0 and plump_env._phase == 0:
                terminated = True
                break

            action_mask = info["action_mask"]
            valid_actions = np.where(action_mask)[0]

            # Safety check: if no valid actions, game must be over
            if len(valid_actions) == 0:
                terminated = True
                break

            # Get opponent action
            if self.opponent_policy is not None:
                flat_obs = self._flatten_obs(obs_dict)
                action = self.opponent_policy(flat_obs, action_mask)
            else:
                # Random valid action
                action = plump_env.np_random.choice(valid_actions)

            obs_dict, reward, step_terminated, truncated, info = self.env.step(action)

            # Accumulate reward for player 0 (the agent) even when opponents act
            # This captures the agent's round reward when opponents finish the round
            total_reward += float(reward)

            if step_terminated or truncated:
                terminated = True
                break

        return obs_dict, info, total_reward, terminated

    def _flatten_obs(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Flatten Dict observation to a single vector."""
        plump_env: PlumpEnv = self.env  # type: ignore
        n_players = plump_env.n_players

        # Extract components
        hand = obs_dict["hand"].astype(np.float32)  # 52
        current_trick = obs_dict["current_trick"].astype(np.float32)  # n_players
        played_cards = obs_dict["played_cards"].astype(np.float32)  # 52
        bids = obs_dict["bids"].astype(np.float32)  # n_players
        total_bids = obs_dict["total_bids"].astype(np.float32)  # 1
        blocked_bid = obs_dict["blocked_bid"].astype(np.float32)  # 1
        tricks_won = obs_dict["tricks_won"].astype(np.float32)  # n_players
        hand_size = np.array([obs_dict["hand_size"]], dtype=np.float32)  # 1
        trump_suit = np.array([obs_dict["trump_suit"]], dtype=np.float32)  # 1
        led_suit = np.array([obs_dict["led_suit"]], dtype=np.float32)  # 1
        phase = np.array([obs_dict["phase"]], dtype=np.float32)  # 1
        scores = obs_dict["scores"].astype(np.float32)  # n_players
        dealer = np.array([obs_dict["dealer"]], dtype=np.float32)  # 1
        current_player = np.array([obs_dict["current_player"]], dtype=np.float32)  # 1

        if self.normalize_obs:
            # Normalize to [0, 1] range
            # hand, played_cards already in [0, 1]
            current_trick = (current_trick + 1) / N_CARDS  # -1 to 51 -> 0 to 1
            bids = (bids + 1) / (MAX_HAND_SIZE + 1)  # -1 to 10 -> 0 to 1
            total_bids = total_bids / (MAX_HAND_SIZE * n_players)
            blocked_bid = (blocked_bid + 1) / (MAX_HAND_SIZE + 1)
            tricks_won = tricks_won / MAX_HAND_SIZE
            hand_size = hand_size / MAX_HAND_SIZE
            trump_suit = trump_suit / 4  # 0-4 -> 0-1
            led_suit = led_suit / 4
            phase = phase  # Already 0 or 1
            scores = scores / 200  # Assume max ~200 points
            dealer = dealer / (n_players - 1) if n_players > 1 else dealer
            current_player = (
                current_player / (n_players - 1) if n_players > 1 else current_player
            )

        obs = np.concatenate(
            [
                hand,
                current_trick,
                played_cards,
                bids,
                total_bids,
                blocked_bid,
                tricks_won,
                hand_size,
                trump_suit,
                led_suit,
                phase,
                scores,
                dealer,
                current_player,
            ]
        )

        return obs

    def action_masks(self) -> np.ndarray:
        """Return the current action mask."""
        return self._action_mask.copy()
