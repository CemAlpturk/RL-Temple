from typing import Callable, Any
import gymnasium as gym
from gymnasium.wrappers import TransformObservation, GrayscaleObservation
import numpy as np
import ale_py
import torch

from rl_temple.trainer import Trainer
from rl_temple.agents.vpg_agent import VPGAgent
from rl_temple.models.actor_critic import MLPActorCritic, CNNActorCritic
from rl_temple.runners import OnPolicyRunner

gym.register_envs(ale_py)

ENV_NAME = "LunarLander-v3"
DEVICE = "cpu"


def get_env_fn(
    env_name: str,
    env_args: dict[str, Any] = {},
    render_mode: str | None = None,
) -> Callable[[], gym.Env]:

    def env_fn() -> gym.Env:
        env = gym.make(env_name, render_mode=render_mode, **env_args)
        # env = GrayscaleObservation(env)
        # env = TransformObservation(env, lambda x: x[np.newaxis], None)
        return env

    return env_fn


env_fn = get_env_fn(ENV_NAME)
eval_env_fn = get_env_fn(ENV_NAME, render_mode="rgb_array")

env = env_fn()

model = MLPActorCritic(
    observation_space=env.observation_space,
    action_space=env.action_space,
    hidden_sizes=[64, 64],
)

# conv_layers = [
#     {
#         "out_channels": 16,
#         "kernel_size": 3,
#         "padding": 1,
#         "batch_norm": True,
#         "pooling": {
#             "type": "max",
#             "kernel_size": 2,
#         },
#     },
#     {
#         "out_channels": 32,
#         "kernel_size": 3,
#         "padding": 1,
#         "batch_norm": True,
#         "pooling": {
#             "type": "max",
#             "kernel_size": 2,
#         },
#     },
# ]
# fc_layers = [128, 64]

# model = CNNActorCritic(
#     observation_space=env.observation_space,
#     action_space=env.action_space,
#     conv_layers=conv_layers,
#     fc_layers=fc_layers,
#     dropout=0.5,
# )

agent = VPGAgent(
    model=model,
    device=torch.device(DEVICE),
)

runner = OnPolicyRunner(agent, env)
trainer = Trainer(
    runner,
    env_fn,
    num_episodes=10000,
    eval_interval=100,
    eval_env_fn=eval_env_fn,
    render_interval=100,
)
trainer.train()

env = gym.make(ENV_NAME, render_mode="human")
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
