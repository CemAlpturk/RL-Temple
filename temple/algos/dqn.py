import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import gymnasium as gym

import random
from collections import deque
from typing import NamedTuple, Callable
from copy import deepcopy

from tqdm import tqdm

# from temple.models import make_model
from temple.logging import TensorboardLogger


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class DQNPolicy(nn.Module):

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.target = deepcopy(model)

        self.model.train()
        self.target.eval()

        self.device = "cpu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @torch.no_grad
    def choose_action(self, state: np.ndarray) -> int:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        return self.model(state).argmax().item()

    @torch.no_grad
    def target_q_values(self, state: torch.Tensor) -> torch.Tensor:
        return self.target(state).max(1)[0].unsqueeze(1)

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse=recurse)

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def train(self):
        self.model.train()
        # self.target.train()

    def eval(self):
        self.model.eval()
        # self.target.eval()

    def to(self, device: torch.device):
        self.model.to(device)
        self.target.to(device)
        self.device = device
        return self

    def save(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "target": self.target.state_dict(),
        }

    def load(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.target.load_state_dict(state["target"])


class DQNAgent:
    def __init__(
        self,
        env_fn: Callable[[str | None], gym.Env],
        # model_config: dict[str, Any],
        model: nn.Module,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_delta: float = 1e-6,
        epsilon_min: float = 0.01,
        batch_size: int = 64,
        memory_size: int = 10000,
        target_update: int = 1000,
        max_steps_per_episode: int = 1000,
        n_episodes: int = 1000,
        n_eval_episodes: int = 10,
        eval_interval: int = 100,
        train_per_step: int = 1,
        record_env_interval: int = 10000,
        log_dir: str | None = None,
    ) -> None:
        self.env_fn = env_fn
        env: gym.Env = env_fn()
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        env.close()

        # self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.policy = DQNPolicy(model).to(self.device)
        self.policy.update_target()
        self.policy.train()
        self.target_update = target_update

        self.gamma = gamma

        # TODO: Epsilon scheduler
        self.epsilon = epsilon
        self.epsilon_delta = epsilon_delta
        self.epsilon_min = epsilon_min

        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        # TODO: Learning rate scheduler
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.max_steps_per_episode = max_steps_per_episode
        self.n_episodes = n_episodes
        self.n_eval_episodes = n_eval_episodes
        self.train_per_step = train_per_step
        self.eval_interval = eval_interval
        self.record_env_interval = record_env_interval
        self.global_step = 0

        # Logging
        self.logger = TensorboardLogger(log_dir)

    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        return self.policy.choose_action(state)

    def learn(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(
            self.device
        )
        action_batch = (
            torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1).to(self.device)
        )
        reward_batch = (
            torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        )
        next_state_batch = torch.tensor(
            np.array(batch.next_state), dtype=torch.float32
        ).to(self.device)
        done_batch = (
            torch.tensor(batch.done, dtype=torch.bool).unsqueeze(1).to(self.device)
        )

        q_values = self.policy(state_batch).gather(1, action_batch)

        next_q_values = self.policy.target_q_values(next_state_batch)
        target_q_values = reward_batch + self.gamma * next_q_values * (~done_batch)

        loss = nn.functional.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_delta
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def train(self):
        self.policy.train()

        env = self.env_fn()

        pbar = tqdm(total=self.n_episodes, desc="Training")
        for episode in range(self.n_episodes):
            episode_reward = 0
            state, _ = env.reset()
            for step in range(self.max_steps_per_episode):
                act = self.choose_action(state)
                next_state, reward, done, _, _ = env.step(act)
                self.memory.append(Transition(state, act, reward, next_state, done))

                if self.global_step % self.train_per_step == 0:
                    self.learn()

                self.global_step += 1
                episode_reward += reward

                if self.global_step % self.target_update == 0:
                    self.policy.update_target()

                if self.global_step % self.eval_interval == 0:
                    eval_rewards, eval_steps = self.eval()

                    # Log evaluation stats
                    self.logger.log_scalar(
                        tag="eval/mean_reward",
                        scalar_value=np.mean(eval_rewards),
                        global_step=self.global_step,
                    )
                    self.logger.log_scalar(
                        tag="eval/mean_steps",
                        scalar_value=np.mean(eval_steps),
                        global_step=self.global_step,
                    )

                if self.global_step % self.record_env_interval == 0:
                    self.record_episode()

                if done:
                    # Log episode stats
                    self.logger.log_scalar(
                        tag="train/episode_reward",
                        scalar_value=episode_reward,
                        global_step=self.global_step,
                    )
                    self.logger.log_scalar(
                        tag="train/episode_steps",
                        scalar_value=step,
                        global_step=self.global_step,
                    )
                    self.logger.log_scalar(
                        tag="train/epsilon",
                        scalar_value=self.epsilon,
                        global_step=self.global_step,
                    )
                    self.logger.log_scalar(
                        tag="train/episode",
                        scalar_value=episode,
                        global_step=self.global_step,
                    )

                    break

                state = next_state

            desc = f"Episode reward: {episode_reward:.4f}, Episode steps: {step}"
            pbar.set_description(desc)
            pbar.update(1)

        pbar.close()
        env.close()
        self.logger.close()

    def eval(self):
        env = self.env_fn()
        self.policy.eval()
        episode_rewards = []
        episode_steps = []
        for _ in range(self.n_eval_episodes):
            state, _ = env.reset()
            episode_reward = 0
            for step in range(self.max_steps_per_episode):
                act = self.choose_action(state, training=False)
                next_state, reward, done, _, _ = env.step(act)
                episode_reward += reward
                if done:
                    break
                state = next_state
            episode_rewards.append(episode_reward)
            episode_steps.append(step)

        env.close()
        return episode_rewards, episode_steps

    def record_episode(self) -> None:
        env: gym.Env = self.env_fn(render_mode="rgb_array")
        self.policy.eval()
        reward = 0
        state, _ = env.reset()
        frames = [env.render()]
        for step in range(self.max_steps_per_episode):
            action = self.choose_action(state, training=False)
            next_state, r, done, _, _ = env.step(action)
            reward += r
            frames.append(env.render())
            if done:
                break
            state = next_state

        env.close()
        # Convert to (N, T, C, H, W)
        frames = (
            torch.tensor(np.array(frames), dtype=torch.uint8).transpose(1, 3).unsqueeze(0)
        )
        self.logger.log_video(
            tag="episode",
            vid_tensor=frames,
            global_step=self.global_step,
            fps=4,
        )

    def save(self) -> dict:
        return self.policy.save()

    def load(self, state: dict) -> None:
        self.policy.load(state)
