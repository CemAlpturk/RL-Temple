import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random


class SnakeEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size: int = 10):
        super().__init__()

        # Grid size
        self.grid_size = grid_size

        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # 0: left, 1: up, 2: right, 3: down
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.grid_size, self.grid_size), dtype=np.int32
        )

        self.snake = None
        self.food = None
        self.done = None
        self.direction = None

        # Initialize pygame
        # TODO: Move this to the render method
        pygame.init()
        self.screen = pygame.display.set_mode((500, 500))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self, **kwargs):
        # Initialize snake in the middle of the grid
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = random.choice([0, 1, 2, 3])
        self.food = self._place_food()
        self.done = False
        return self._get_obs(), {}

    def _place_food(self):
        # FIXME: Potential infinite loop
        while True:
            food = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            if food not in self.snake:
                return food

    def _get_obs(self):
        obs = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for x, y in self.snake:
            obs[x, y] = 1
        obs[self.food[1], self.food[0]] = 2

        return obs

    # def step(self, action):
    #     if self.done:
    #         raise Exception(
    #             "Cannot call step() on a finished game. Please reset() the environment."
    #         )

    #     # Map actions to direction changes
    #     if action == 0:  # left
    #         self.direction = 0 if self.direction != 2 else self.direction
    #     elif action == 1:  # up
    #         self.direction = 1 if self.direction != 3 else self.direction
    #     elif action == 2:  # right
    #         self.direction = 2 if self.direction != 0 else self.direction
    #     elif action == 3:  # down
    #         self.direction = 3 if self.direction != 1 else self.direction

    #     # Move snake
    #     head_x, head_y = self.snake[0]
    #     if self.direction == 0:
    #         head_x = (head_x - 1) % self.grid_size
    #     elif self.direction == 1:
    #         head_y = (head_y - 1) % self.grid_size
    #     elif self.direction == 2:
    #         head_x = (head_x + 1) % self.grid_size
    #     elif self.direction == 3:
    #         head_y = (head_y + 1) % self.grid_size

    #     # Check for collisions
    #     if (head_x, head_y) in self.snake:
    #         self.done = True
    #         reward = -1
    #     else:
    #         self.snake.insert(0, (head_x, head_y))
    #         if (head_x, head_y) == self.food:
    #             reward = 1
    #             self.food = self._place_food()
    #         else:
    #             reward = 0
    #             self.snake.pop()

    #     obs = self._get_obs()
    #     return obs, reward, self.done, False, {}
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
            not (0 <= head_x < self.grid_size and 0 <= head_y < self.grid_size)
            or (head_x, head_y) in self.snake
        ):
            self.done = True
            reward = -1
        else:
            self.snake.insert(0, (head_x, head_y))
            if (head_x, head_y) == self.food:
                reward = 1
                self.food = self._place_food()
            else:
                reward = 0
                self.snake.pop()

        obs = self._get_obs()
        return obs, reward, self.done, False, {}

    def render(self, mode="human"):
        if mode != "human":
            raise NotImplementedError()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Calculate cell size
        cell_size = self.screen.get_width() // self.grid_size

        self.screen.fill((0, 0, 0))

        # Draw snake
        for x, y in self.snake:
            pygame.draw.rect(
                self.screen,
                (0, 255, 0),
                pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size),
            )

        # Draw food
        pygame.draw.rect(
            self.screen,
            (255, 0, 0),
            pygame.Rect(
                self.food[0] * cell_size, self.food[1] * cell_size, cell_size, cell_size
            ),
        )

        pygame.display.flip()
        self.clock.tick(10)  # Set the game speed to 10 frames per second

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = SnakeEnv(grid_size=10)
    env.reset()

    for _ in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, done, _, info = env.step(action)
        env.render()
        if done:
            env.reset()

    env.close()
