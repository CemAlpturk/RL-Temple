from typing import Callable, Any
import argparse

import gymnasium as gym
import torch

from rl_temple.trainer import Trainer
from rl_temple.agents.ppo_agent import PPOAgent
from rl_temple.models.actor_critic import MLPActorCritic
from rl_temple.runners import OnPolicyRunner

ENV_NAME = "LunarLander-v3"
DEVICE = "cpu"


# Env fn
def get_env_fn(
    env_args: dict[str, Any] = {},
    render_mode: str | None = None,
) -> Callable[[], gym.Env]:
    """
    Returns a function that creates a new environment instance.
    """

    def env_fn() -> gym.Env:
        env = gym.make(ENV_NAME, render_mode=render_mode, **env_args)
        return env

    return env_fn


def main(
    device: str,
    hidden_sizes: list[int],
    pi_lr: float,
    vf_lr: float,
    gamma: float,
    lam: float,
    batch_size: int,
    epochs: int,
    n_episodes_per_train: int,
    clip_ratio: float,
    target_kl: float,
    total_episodes: int,
    max_steps_per_episode: int,
    eval_interval: int,
    eval_episodes: int,
) -> None:
    # Create environment
    env_fn = get_env_fn()
    eval_env_fn = get_env_fn(render_mode="rgb_array")

    env = env_fn()

    # Create model
    model = MLPActorCritic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_sizes=hidden_sizes,
        activation="relu",
    )

    # Create agent
    agent = PPOAgent(
        model=model,
        device=torch.device(device),
        pi_lr=pi_lr,
        vf_lr=vf_lr,
        gamma=gamma,
        lam=lam,
        batch_size=batch_size,
        epochs=epochs,
        n_episodes=n_episodes_per_train,
        clip_ratio=clip_ratio,
        target_kl=target_kl,
    )

    # Runner and trainer
    runner = OnPolicyRunner(
        agent=agent,
        env=env,
    )
    trainer = Trainer(
        runner=runner,
        env_fn=env_fn,
        num_episodes=total_episodes,
        max_steps_per_episode=max_steps_per_episode,
        eval_env_fn=eval_env_fn,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        render_interval=eval_interval,
    )

    trainer.train()

    # Training finished
    env = get_env_fn(render_mode="human")()
    while True:
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(state, explore=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

        print(f"Total reward: {total_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lunar Lander VPG Example")
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help="Device to use",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[64, 64],
        help="Hidden sizes for the MLP",
    )
    parser.add_argument(
        "--pi_lr",
        type=float,
        default=3e-4,
        help="Learning rate for the policy network",
    )
    parser.add_argument(
        "--vf_lr",
        type=float,
        default=3e-4,
        help="Learning rate for the value network",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.97,
        help="Lambda for GAE",
    )
    parser.add_argument(
        "--clip_ratio",
        type=float,
        default=0.2,
        help="Clipping ratio for PPO",
    )
    parser.add_argument(
        "--target_kl",
        type=float,
        default=0.01,
        help="Target KL divergence for PPO",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--n_episodes_per_train",
        type=int,
        default=100,
        help="Number of episodes to train on per iteration",
    )
    parser.add_argument(
        "--total_episodes",
        type=int,
        default=50000,
        help="Number of episodes to train on",
    )
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=None,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1000,
        help="Interval for evaluation",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=5,
        help="Number of episodes for evaluation",
    )
    args = parser.parse_args()
    main(
        device=args.device,
        hidden_sizes=args.hidden_sizes,
        pi_lr=args.pi_lr,
        vf_lr=args.vf_lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lam=args.lam,
        n_episodes_per_train=args.n_episodes_per_train,
        clip_ratio=args.clip_ratio,
        target_kl=args.target_kl,
        total_episodes=args.total_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
    )
