from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode: str | None = None) -> None:
        self.size = 10  # Size of the grid
        self.window_size = 500  # Size of the PyGame window

        # Action space
        # 0: left, 1: up, 2: right, 3: down
        self.action_space = spaces.Discrete(4)

        # Observation space is a 2D grid of size (size, size)
        self.observation_space = spaces.Box(
            low=0,
            high=4,
            shape=(self.size, self.size),
            dtype=np.uint8,
        )

        # Reward range
        self.reward_range = (-1, 10)

        # Maximum number of steps before the game is terminated
        self._max_steps = 1000

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Rendering objects
        self.window = None
        self.clock = None

        # Render colors
        self._wall_color = (150, 75, 0)
        self._snake_color = (0, 255, 0)
        self._head_color = (0, 0, 0)
        self._food_color = (255, 0, 0)

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.size, self.size), dtype=np.uint8)
        for x, y in self._snake:
            obs[x, y] = 2
        obs[self._snake[0][0], self._snake[0][1]] = 1  # Head of the snake
        obs[self._food[1], self._food[0]] = 3

        # Wall piece
        obs[0, :] = 4
        obs[-1, :] = 4
        obs[:, 0] = 4
        obs[:, -1] = 4

        return obs

    def _get_info(self) -> dict[str, Any]:
        return {}

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # Reset with a new seed
        super().reset(seed=seed)

        # Initialize snake in the middle of the grid
        head = (self.size // 2, self.size // 2)
        self._snake = [head, (head[0] - 1, head[1]), (head[0] - 2, head[1])]
        self._direction = 2
        self._food = self._place_food()

        self._terminated = False
        self._it = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _place_food(self) -> tuple[int, int]:
        # FIXME: Potential infinite loop
        while True:
            # Place food at a random location between walls
            food = (
                random.randint(1, self.size - 2),
                random.randint(1, self.size - 2),
            )
            if food not in self._snake:
                return food

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._terminated:
            raise Exception(
                "Cannot call step() on a finished game. Please reset() the environment."
            )

        # Map actions to direction changes
        if action == 0:  # left
            self._direction = 0 if self._direction != 2 else self._direction
        elif action == 1:  # up
            self._direction = 1 if self._direction != 3 else self._direction
        elif action == 2:  # right
            self._direction = 2 if self._direction != 0 else self._direction
        elif action == 3:  # down
            self._direction = 3 if self._direction != 1 else self._direction

        # Move snake
        head_x, head_y = self._snake[0]
        if self._direction == 0:
            head_x -= 1
        elif self._direction == 1:
            head_y -= 1
        elif self._direction == 2:
            head_x += 1
        elif self._direction == 3:
            head_y += 1

        # Check for collisions
        if (
            not (0 < head_x < self.size - 1 and 0 < head_y < self.size - 1)
            or (head_x, head_y) in self._snake
        ):
            self._terminated = True
            reward = -1
        else:
            self._snake.insert(0, (head_x, head_y))
            if (head_x, head_y) == self._food:
                reward = 10
                self._food = self._place_food()
            else:
                reward = 0
                self._snake.pop()

        self._it += 1
        truncated = False
        if self._it >= self._max_steps:
            truncated = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, self._terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> np.ndarray | None:
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Size of a single grid square in pixels
        pix_square_size = self.window_size / self.size

        # Draw walls
        pygame.draw.rect(
            canvas,
            self._wall_color,
            pygame.Rect(
                0,
                0,
                self.size * pix_square_size,
                pix_square_size,
            ),
        )
        pygame.draw.rect(
            canvas,
            self._wall_color,
            pygame.Rect(
                0,
                self.size * pix_square_size - pix_square_size,
                self.size * pix_square_size,
                pix_square_size,
            ),
        )
        pygame.draw.rect(
            canvas,
            self._wall_color,
            pygame.Rect(
                0,
                0,
                pix_square_size,
                self.size * pix_square_size,
            ),
        )
        pygame.draw.rect(
            canvas,
            self._wall_color,
            pygame.Rect(
                self.size * pix_square_size - pix_square_size,
                0,
                pix_square_size,
                self.size * pix_square_size,
            ),
        )

        # Draw snake
        for i, (x, y) in enumerate(self._snake):
            pygame.draw.rect(
                canvas,
                self._snake_color if i != 0 else self._head_color,
                pygame.Rect(
                    x * pix_square_size,
                    y * pix_square_size,
                    pix_square_size,
                    pix_square_size,
                ),
            )

        # Draw food
        pygame.draw.rect(
            canvas,
            self._food_color,
            pygame.Rect(
                self._food[0] * pix_square_size,
                self._food[1] * pix_square_size,
                pix_square_size,
                pix_square_size,
            ),
        )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":

    env = SnakeEnv(render_mode="human")
    env.reset()

    for _ in range(50):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, _, info = env.step(action)
        if terminated:
            env.reset()

    env.close()
