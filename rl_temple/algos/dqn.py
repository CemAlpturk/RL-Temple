import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import gymnasium as gym

import random
from collections import deque
from typing import NamedTuple
from copy import deepcopy

from tqdm import tqdm

# from temple.models import make_model
from rl_temple.logging import TensorboardLogger


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
        env: gym.Env,
        model: nn.Module,
        test_env: gym.Env | None = None,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_delta: float = 1e-6,
        epsilon_min: float = 0.01,
        batch_size: int = 64,
        memory_size: int = 10000,
        target_update: int = 1000,
        max_steps: int = int(1e6),
        n_eval_episodes: int = 10,
        eval_interval: int = 100,
        train_per_step: int = 1,
        record_env_interval: int = 10000,
        log_dir: str | None = None,
    ) -> None:
        self.env = env

        assert isinstance(
            env.action_space, gym.spaces.Discrete
        ), "Only Discrete action spaces are supported"
        self.action_size = env.action_space.n

        if test_env is None:
            self.test_env = deepcopy(env)
        else:
            self.test_env = test_env

        self.device = torch.device("cpu")
        self.policy = DQNPolicy(model).to(self.device)
        self.policy.update_target()
        self.policy.train()
        self.target_update = target_update

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_delta = epsilon_delta
        self.epsilon_min = epsilon_min

        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        # TODO: Learning rate scheduler
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.max_steps = max_steps
        self.n_eval_episodes = n_eval_episodes
        self.train_per_step = train_per_step
        self.eval_interval = eval_interval
        self.record_env_interval = record_env_interval

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

        episode_reward = 0
        episode_steps = 0
        n_episodes = 0
        state, _ = self.env.reset()
        pbar = tqdm(total=self.max_steps, desc="Training")
        for step in range(self.max_steps):
            act = self.choose_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(act)
            self.memory.append(Transition(state, act, reward, next_state, terminated))
            done = terminated or truncated

            state = next_state

            if step % self.train_per_step == 0:
                self.learn()

            episode_reward += reward
            episode_steps += 1

            if step % self.target_update == 0:
                self.policy.update_target()

            if step % self.eval_interval == 0:
                eval_rewards, eval_steps = self.eval()

                # Log evaluation stats
                self.logger.log_scalar(
                    tag="eval/mean_reward",
                    scalar_value=np.mean(eval_rewards),
                    global_step=step,
                )
                self.logger.log_scalar(
                    tag="eval/mean_steps",
                    scalar_value=np.mean(eval_steps),
                    global_step=step,
                )
                desc = f"Eval mean reward: {np.mean(eval_rewards):.4f}, Eval median steps: {np.median(eval_steps)}"
                pbar.set_description(desc)

            if step % self.record_env_interval == 0:
                self.record_episode(step)

            if done:
                n_episodes += 1

                # Log episode stats
                self.logger.log_scalar(
                    tag="train/episode_reward",
                    scalar_value=episode_reward,
                    global_step=step,
                )
                self.logger.log_scalar(
                    tag="train/episode_steps",
                    scalar_value=episode_steps,
                    global_step=step,
                )
                self.logger.log_scalar(
                    tag="train/epsilon",
                    scalar_value=self.epsilon,
                    global_step=step,
                )
                self.logger.log_scalar(
                    tag="train/episode",
                    scalar_value=n_episodes,
                    global_step=step,
                )

                # Reset episode stats
                episode_reward = 0
                episode_steps = 0
                state, _ = self.env.reset()

            pbar.update(1)

        pbar.close()
        self.env.close()
        self.test_env.close()
        self.logger.close()

    def eval(self):
        self.policy.eval()
        episode_rewards = []
        episode_steps = []
        for _ in range(self.n_eval_episodes):
            state, _ = self.test_env.reset()
            episode_reward = 0
            done = False
            step = 0
            while not done:
                act = self.choose_action(state, training=False)
                next_state, reward, terminated, truncated, _ = self.test_env.step(act)
                done = terminated or truncated
                episode_reward += reward
                state = next_state
                step += 1
            episode_rewards.append(episode_reward)
            episode_steps.append(step)

        return episode_rewards, episode_steps

    def record_episode(self, step: int) -> None:
        self.policy.eval()
        reward = 0
        state, _ = self.test_env.reset()
        done = False
        while not done:
            action = self.choose_action(state, training=False)
            next_state, r, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            reward += r
            state = next_state

        frames = self.test_env.render()  # (T, H, W, C)

        # Convert to (N, T, C, H, W)
        frames = torch.tensor(
            np.array(frames).transpose(0, 3, 1, 2),
            dtype=torch.uint8,
        ).unsqueeze(0)
        self.logger.log_video(
            tag="episode",
            vid_tensor=frames,
            global_step=step,
            fps=self.test_env.metadata["render_fps"],
        )

    def save(self) -> dict:
        return self.policy.save()

    def load(self, state: dict) -> None:
        self.policy.load(state)
