from typing import Callable, Any
import gymnasium as gym

import numpy as np
import ale_py
import torch

from rl_temple.trainer import Trainer
from rl_temple.agents.vpg_agent import VPGAgent
from rl_temple.models.actor_critic import MLPActorCritic
from rl_temple.runners import OnPolicyRunner


gym.register_envs(ale_py)

ENV_NAME = "ALE/Pong-v5"
DEVICE = "cpu"


def get_env_fn(
    env_name: str,
    env_args: dict[str, Any] = {},
    render_mode: str | None = None,
) -> Callable[[], gym.Env]:

    def process(obs: np.ndarray) -> np.ndarray:
        obs = obs[:, 35:195]
        obs = obs[:, ::2, ::2, 0]
        obs[obs == 144] = 0
        obs[obs == 109] = 0
        obs[obs != 0] = 1
        obs = obs.astype(np.float32)

        return obs.ravel()

    def env_fn() -> gym.Env:
        args = {
            **env_args,
        }
        env = gym.make(env_name, render_mode=render_mode, **args)
        env = gym.wrappers.FrameStackObservation(env, 2)

        obs_space = gym.spaces.Box(low=0, high=1, shape=(2 * 80 * 80,), dtype=np.float32)
        env = gym.wrappers.TransformObservation(env, process, obs_space)
        return env

    return env_fn


env_fn = get_env_fn(ENV_NAME)
eval_env_fn = get_env_fn(ENV_NAME, render_mode="rgb_array")

env = env_fn()

model = MLPActorCritic(
    observation_space=env.observation_space,
    action_space=env.action_space,
    hidden_sizes=[200],
    activation="relu",
)

agent = VPGAgent(
    model=model,
    device=torch.device(DEVICE),
    pi_lr=3e-4,
    vf_lr=1e-3,
    gamma=0.99,
    batch_size=10,
)

runner = OnPolicyRunner(agent, env)
trainer = Trainer(
    runner,
    env_fn,
    num_episodes=10000,
    eval_interval=50,
    eval_env_fn=eval_env_fn,
    render_interval=50,
    max_steps_per_episode=None,
)
trainer.train()

env = get_env_fn(ENV_NAME, render_mode="human")()
while True:

    state, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = agent.select_action(state, explore=False)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)

    print(f"Total Reward: {total_reward}")
