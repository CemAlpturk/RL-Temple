from typing import Callable, Optional

from tqdm import tqdm

import gymnasium as gym

from rl_temple.runners import AgentRunner
from rl_temple.logging.tensorboard_logger import TensorboardLogger


class Trainer:

    def __init__(
        self,
        runner: AgentRunner,
        env_fn: Callable[[], gym.Env],
        num_episodes: int = 500,
        max_steps_per_episode: int = 500,
        eval_env_fn: Optional[Callable[[], gym.Env]] = None,
        eval_interval: int = 10,
        eval_episodes: int = 5,
    ) -> None:
        self.episode = 0
        self.runner = runner
        self.agent = self.runner.agent
        self.env = env_fn()
        self.num_episodes = num_episodes
        self.max_steps = max_steps_per_episode
        self.eval_env = eval_env_fn() if eval_env_fn else None
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes

        self.logger = TensorboardLogger()

    def train(self):

        for episode in (pbar := tqdm(range(1, self.num_episodes + 1))):
            self.episode = episode
            total_reward = self.runner.run_episode(self.max_steps)

            pbar_str = f"Reward: {total_reward:.2f}"

            if self.eval_env and episode % self.eval_interval == 0:
                avg_reward = self.evaluate()
                pbar_str += f" Eval: {avg_reward:.2f}"

            pbar.set_description(pbar_str)

            # Logging
            self.logger.log_scalar(
                tag="train/episode_reward",
                scalar_value=total_reward,
                global_step=self.episode,
            )

    def evaluate(self) -> float:
        if not self.eval_env:
            return 0.0

        eval_runner = type(self.runner)(self.agent, self.eval_env)

        rewards = [
            eval_runner.run_episode(self.max_steps, explore=False)
            for _ in range(self.eval_episodes)
        ]

        mean_reward = sum(rewards) / len(rewards)

        self.logger.log_scalar(
            tag="eval/mean_reward",
            scalar_value=mean_reward,
            global_step=self.episode,
        )

        return mean_reward
