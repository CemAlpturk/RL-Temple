from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces


class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self) -> None:
        super().__init__()

        # 3x3 board
        self.board = [" "] * 9
        self.current_player = "X"
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.MultiDiscrete([3] * 9)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        self.board = [" "] * 9
        self.current_player = "X"
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, int, bool, bool, dict]:
        if self.board[action] != " ":
            return self._get_obs(), -10, True, False, {}

        self.board[action] = self.current_player
        winner: Optional[str] = self._check_winner()
        done: bool = winner is not None
        reward: int = 1 if winner == "X" else -1 if winner == "O" else 0
        self.current_player = "O" if self.current_player == "X" else "X"
        return self._get_obs(), reward, done, False, {}

    def render(self) -> None:
        print("\n".join([" ".join(self.board[i : i + 3]) for i in range(0, 9, 3)]))
        print()

    def _get_obs(self) -> np.ndarray:
        mapping: dict[str, int] = {"X": 1, "O": 2, " ": 0}
        return np.array([mapping[cell] for cell in self.board])

    def _check_winner(self) -> Optional[str]:
        win_states = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),  # Rows
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),  # Columns
            (0, 4, 8),
            (2, 4, 6),  # Diagonals
        ]
        for i, j, k in win_states:
            if self.board[i] == self.board[j] == self.board[k] and self.board[i] != " ":
                return self.board[i]
        if " " not in self.board:
            return "Draw"
        return None
