import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", None]}

    def __init__(self, render_mode: str | None = None):
        super().__init__()

        # Grid size
        self.grid_size = (10, 10)

        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # 0: left, 1: up, 2: right, 3: down
        self.observation_space = spaces.Box(
            low=0, high=4, shape=self.grid_size, dtype=np.int32
        )

        self.reward_range = (-1, 10)

        self.snake = None
        self.food = None
        self.done = None
        self.direction = None
        # self.prev_states = None

        # Initialize pygame
        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((500, 500))
            pygame.display.set_caption("Snake Game")
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self, **kwargs):
        # Initialize snake in the middle of the grid
        head = (self.grid_size[0] // 2, self.grid_size[1] // 2)
        self.snake = [head, (head[0] - 1, head[1]), (head[0] - 2, head[1])]
        self.diraction = 2
        # self.snake = [(self.grid_size[0] // 2, self.grid_size[1] // 2)]
        # self.direction = random.choice([0, 1, 2, 3])
        self.food = self._place_food()
        self.done = False
        # self.prev_states = np.zeros_like(self.observation_space.low)[None, :].repeat(
        #     4, axis=0
        # )
        return self._get_obs(), {}

    def _place_food(self):
        # FIXME: Potential infinite loop
        while True:
            food = (
                random.randint(1, self.grid_size[0] - 2),
                random.randint(1, self.grid_size[1] - 2),
            )
            if food not in self.snake:
                return food

    def _get_obs(self):
        obs = np.zeros(self.grid_size, dtype=np.int32)
        for x, y in self.snake:
            obs[x, y] = 2
        obs[self.snake[0][0], self.snake[0][1]] = 1  # Head of the snake
        obs[self.food[1], self.food[0]] = 3

        # Wall piece
        obs[0, :] = 4
        obs[-1, :] = 4
        obs[:, 0] = 4
        obs[:, -1] = 4

        return obs
        # self.prev_states = np.roll(self.prev_states, 1, axis=0)
        # self.prev_states[0] = obs

        # return self.prev_states

    def step(self, action):
        if self.done:
            raise Exception(
                "Cannot call step() on a finished game. Please reset() the environment."
            )

        # Map actions to direction changes
        if action == 0:  # left
            self.direction = 0 if self.direction != 2 else self.direction
        elif action == 1:  # up
            self.direction = 1 if self.direction != 3 else self.direction
        elif action == 2:  # right
            self.direction = 2 if self.direction != 0 else self.direction
        elif action == 3:  # down
            self.direction = 3 if self.direction != 1 else self.direction

        # Move snake
        head_x, head_y = self.snake[0]
        if self.direction == 0:
            head_x -= 1
        elif self.direction == 1:
            head_y -= 1
        elif self.direction == 2:
            head_x += 1
        elif self.direction == 3:
            head_y += 1

        # Check for collisions
        if (
            not (
                0 < head_x < self.grid_size[0] - 1 and 0 < head_y < self.grid_size[1] - 1
            )
            or (head_x, head_y) in self.snake
        ):
            self.done = True
            reward = -1
        else:
            self.snake.insert(0, (head_x, head_y))
            if (head_x, head_y) == self.food:
                reward = 10
                self.food = self._place_food()
            else:
                reward = 0
                self.snake.pop()

        obs = self._get_obs()
        return obs, reward, self.done, False, {}

    def render(self) -> np.ndarray | None:

        if self.render_mode is None:
            return

        elif self.render_mode == "human":
            self._render_human()
            return

        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

        else:
            raise ValueError(f"Invalid render mode: {self.render_mode}")

    def _render_rgb_array(self) -> np.ndarray:

        wall_color = np.array([150, 75, 0], dtype=np.uint8)
        snake_color = np.array([0, 255, 0], dtype=np.uint8)
        head_color = np.array([0, 0, 0], dtype=np.uint8)
        food_color = np.array([255, 0, 0], dtype=np.uint8)

        window_size = 500
        cell_size = window_size // self.grid_size[0]
        screen = 255 * np.ones((window_size, window_size, 3), dtype=np.uint8)

        # Draw walls
        screen[0:cell_size, :] = wall_color
        screen[-cell_size:, :] = wall_color
        screen[:, 0:cell_size] = wall_color
        screen[:, -cell_size:] = wall_color

        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            screen[
                x * cell_size : (x + 1) * cell_size, y * cell_size : (y + 1) * cell_size
            ] = (snake_color if i != 0 else head_color)

        # Draw food
        x, y = self.food
        screen[
            x * cell_size : (x + 1) * cell_size, y * cell_size : (y + 1) * cell_size
        ] = food_color

        return np.flip(screen.transpose(1, 0, 2), 0)  # (W, H, C) -> (H, W, C)

    def _render_human(self) -> None:
        # Not sure if this is needed?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Calculate cell size
        cell_size_x = self.screen.get_width() // self.grid_size[0]
        cell_size_y = self.screen.get_height() // self.grid_size[1]

        self.screen.fill((255, 255, 255))

        # Draw walls
        for i in range(self.grid_size[0]):
            pygame.draw.rect(
                self.screen,
                (150, 75, 0),
                pygame.Rect(i * cell_size_x, 0, cell_size_x, cell_size_y),
            )
            pygame.draw.rect(
                self.screen,
                (150, 75, 0),
                pygame.Rect(
                    i * cell_size_x,
                    (self.grid_size[1] - 1) * cell_size_y,
                    cell_size_x,
                    cell_size_y,
                ),
            )

        for i in range(self.grid_size[1]):
            pygame.draw.rect(
                self.screen,
                (150, 75, 0),
                pygame.Rect(0, i * cell_size_y, cell_size_x, cell_size_y),
            )
            pygame.draw.rect(
                self.screen,
                (150, 75, 0),
                pygame.Rect(
                    (self.grid_size[0] - 1) * cell_size_x,
                    i * cell_size_y,
                    cell_size_x,
                    cell_size_y,
                ),
            )

        # Draw snake
        x, y = self.snake[0]
        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            pygame.Rect(x * cell_size_x, y * cell_size_y, cell_size_x, cell_size_y),
        )
        for x, y in self.snake[1:]:
            pygame.draw.rect(
                self.screen,
                (0, 255, 0),
                pygame.Rect(x * cell_size_x, y * cell_size_y, cell_size_x, cell_size_y),
            )

        # Draw food
        pygame.draw.rect(
            self.screen,
            (255, 0, 0),
            pygame.Rect(
                self.food[0] * cell_size_x,
                self.food[1] * cell_size_y,
                cell_size_x,
                cell_size_y,
            ),
        )

        pygame.display.flip()
        self.clock.tick(10)  # Set the game speed to 10 frames per second

    def close(self):
        if self.render_mode == "human":
            pygame.quit()


if __name__ == "__main__":
    # import time

    env = SnakeEnv(render_mode="human")
    env.reset()

    for _ in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, done, _, info = env.step(action)
        env.render()
        # time.sleep(0.5)
        if done:
            env.reset()

    env.close()
