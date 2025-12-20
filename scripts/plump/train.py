"""
Training script for PPO agent on Plump card game.

This script trains a PPO agent to play Plump against random opponents,
with support for self-play using past model checkpoints.
"""

import os
import random
from datetime import datetime
from collections import deque
import copy

import gymnasium as gym  # noqa: F401
from gymnasium import spaces
import torch
import numpy as np
from tqdm import tqdm

from rl_temple.environments.plump import PlumpEnv
from rl_temple.wrappers.plump_wrapper import PlumpWrapper
from rl_temple.agents.ppo_agent import PPOAgent
from rl_temple.models.actor_critic import MLPActorCritic
from rl_temple.logging.wb_logger import WandbLogger


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Environment
    "n_players": 5,  # Total players including agent
    "start_hand_size": 10,  # Start with full game (10 rounds)
    # Model architecture
    "hidden_sizes": (1024, 1024),
    "activation": "relu",
    # PPO hyperparameters
    "gamma": 0.99,  # Discount factor
    "lam": 0.95,  # GAE lambda
    "pi_lr": 3e-4,  # Policy learning rate
    "vf_lr": 1e-3,  # Value function learning rate
    "n_episodes": 1024,  # Episodes to collect before update
    "epochs": 10,  # Training epochs per batch
    "batch_size": 64,  # Minibatch size
    "clip_ratio": 0.2,  # PPO clip ratio
    "target_kl": 0.01,  # Early stopping KL threshold
    # Training
    "total_episodes": 100000,
    "eval_interval": 100,  # Evaluate every N episodes
    "eval_episodes": 20,  # Number of episodes for evaluation
    "save_interval": 500,  # Save checkpoint every N episodes
    "seed": 42,
    # Self-play settings
    "use_self_play": True,  # Whether to use past models as opponents
    "self_play_start_episode": 1000,  # Start self-play after N episodes
    "opponent_update_interval": 1000,  # Update opponent pool every N episodes
    "opponent_pool_size": 5,  # Number of past models to keep
    "random_opponent_prob": 0.2,  # Probability of using random opponent
    # Checkpointing
    "checkpoint_dir": "checkpoints/plump",
    "resume_from": None,  # Path to checkpoint to resume from
    # Logging
    "wandb_project": "plump",
    "wandb_mode": "online",  # "online", "offline", or "disabled"
}


# =============================================================================
# Helper Functions
# =============================================================================


def make_env(
    n_players: int,
    seed: int | None = None,
    opponent_policy: "OpponentPolicy | None" = None,
) -> PlumpWrapper:
    """Create a wrapped Plump environment."""
    env = PlumpEnv(n_players=n_players)
    policy_fn = opponent_policy.get_action if opponent_policy else None
    wrapped_env = PlumpWrapper(env, opponent_policy=policy_fn, normalize_obs=True)
    return wrapped_env


class OpponentPolicy:
    """
    Manages opponent policies for self-play training.

    Supports:
    - Random opponents (for warm-up)
    - Past model checkpoints (for self-play)
    - Mixed policies with configurable randomness
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidden_sizes: tuple[int, ...],
        activation: str,
        pool_size: int = 5,
        random_prob: float = 0.2,
        device: str = "cpu",
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.pool_size = pool_size
        self.random_prob = random_prob
        self.device = device

        # Pool of past models (FIFO queue)
        self.model_pool: deque[MLPActorCritic] = deque(maxlen=pool_size)
        self._rng = np.random.default_rng()

    def add_model(self, model: MLPActorCritic) -> None:
        """Add a copy of the current model to the opponent pool."""
        model_copy = MLPActorCritic(
            observation_space=self.observation_space,
            action_space=self.action_space,
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
        )
        model_copy.load_state_dict(copy.deepcopy(model.state_dict()))
        model_copy.eval()
        model_copy.to(self.device)
        self.model_pool.append(model_copy)

    def get_action(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """
        Get an action from the opponent policy.

        With probability `random_prob`, returns a random valid action.
        Otherwise, samples from a random model in the pool.
        """
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            return 0  # Fallback (shouldn't happen)

        # Random action with some probability, or if pool is empty
        if len(self.model_pool) == 0 or self._rng.random() < self.random_prob:
            return int(self._rng.choice(valid_actions))

        # Sample from a random model in the pool
        model = random.choice(list(self.model_pool))

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            mask_tensor = torch.as_tensor(
                action_mask, dtype=torch.bool, device=self.device
            )

            # Get action from model with action masking
            pi, _ = model.pi(
                obs_tensor.unsqueeze(0), action_mask=mask_tensor.unsqueeze(0)
            )
            action = pi.sample().item()

        return int(action)

    def load_from_checkpoints(self, checkpoint_paths: list[str]) -> None:
        """Load models from checkpoint files into the pool."""
        for path in checkpoint_paths:
            if os.path.exists(path):
                model = MLPActorCritic(
                    observation_space=self.observation_space,
                    action_space=self.action_space,
                    hidden_sizes=self.hidden_sizes,
                    activation=self.activation,
                )
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.eval()
                model.to(self.device)
                self.model_pool.append(model)
                print(f"Loaded opponent model from {path}")


def save_checkpoint(
    model: MLPActorCritic,
    agent: PPOAgent,
    episode: int,
    total_steps: int,
    checkpoint_dir: str,
    config: dict,
) -> str:
    """
    Save a training checkpoint.

    Returns:
        Path to the saved checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode}.pt")

    checkpoint = {
        "episode": episode,
        "total_steps": total_steps,
        "model_state_dict": model.state_dict(),
        "config": config,
    }

    torch.save(checkpoint, checkpoint_path)

    # Also save a "latest" checkpoint for easy resuming
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: MLPActorCritic,
) -> tuple[int, int]:
    """
    Load a training checkpoint.

    Returns:
        Tuple of (episode number, total_steps) to resume from.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    total_steps = checkpoint.get("total_steps", 0)  # Backwards compatibility
    return checkpoint["episode"], total_steps


def evaluate(
    agent: PPOAgent,
    env: PlumpWrapper,
    n_episodes: int,
) -> dict[str, float]:
    """
    Evaluate the agent for n_episodes.

    Returns:
        Dictionary with evaluation metrics.
    """
    total_rewards: list[float] = []
    wins = 0

    for _ in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action_mask = env.action_masks()
            # select_action with explore=False, return_info=False returns just the action
            action_result = agent.select_action(
                obs, explore=False, action_mask=action_mask
            )
            action = int(action_result)  # type: ignore
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

        # Check if agent won (highest score)
        plump_env: PlumpEnv = env.env  # type: ignore
        scores = plump_env._scores
        if int(np.argmax(scores)) == 0:
            wins += 1

    return {
        "eval/mean_reward": float(np.mean(total_rewards)),
        "eval/std_reward": float(np.std(total_rewards)),
        "eval/win_rate": float(wins / n_episodes),
    }


def train_episode(
    agent: PPOAgent,
    env: PlumpWrapper,
) -> tuple[dict[str, float], int]:
    """
    Run a single training episode.

    Returns:
        Tuple of (episode statistics, episode length in steps).
    """
    obs, info = env.reset()
    episode_reward = 0.0
    episode_length = 0
    done = False

    while not done:
        action_mask = env.action_masks()

        # Safety check: if no valid actions, episode should be done
        if not action_mask.any():
            break

        action, (log_prob, value) = agent.select_action(
            obs, explore=True, return_info=True, action_mask=action_mask
        )

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition
        agent.remember(
            state=obs,
            action=action,
            reward=reward,
            next_state=next_obs,
            done=done,
            log_prob=log_prob,
            value=value,
            action_mask=action_mask,
        )

        obs = next_obs
        episode_reward += reward
        episode_length += 1

    # End episode and potentially update
    update_stats = agent.end_episode()

    stats = {
        "train/episode_reward": episode_reward,
        "train/episode_length": episode_length,
    }

    if update_stats:
        stats.update({f"train/{k}": v for k, v in update_stats.items()})

    return stats, episode_length


# =============================================================================
# Main Training Loop
# =============================================================================


def main():
    """Main training function."""
    # Set seeds
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    # Create timestamped checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(CONFIG["checkpoint_dir"], timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)
    tqdm.write(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Create base environment to get spaces
    base_env = make_env(CONFIG["n_players"], seed=CONFIG["seed"])

    # Create opponent policy for self-play (if enabled)
    opponent_policy = None
    if CONFIG["use_self_play"]:
        opponent_policy = OpponentPolicy(
            observation_space=base_env.observation_space,  # type: ignore[arg-type]
            action_space=base_env.action_space,  # type: ignore[arg-type]
            hidden_sizes=CONFIG["hidden_sizes"],
            activation=CONFIG["activation"],
            pool_size=CONFIG["opponent_pool_size"],
            random_prob=CONFIG["random_opponent_prob"],
        )
        tqdm.write(f"Self-play enabled with pool size {CONFIG['opponent_pool_size']}")

    # Create environments
    env = make_env(
        CONFIG["n_players"], seed=CONFIG["seed"], opponent_policy=opponent_policy
    )
    eval_env = make_env(
        CONFIG["n_players"], seed=CONFIG["seed"] + 1000
    )  # Eval always vs random

    # Create model
    model = MLPActorCritic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_sizes=CONFIG["hidden_sizes"],
        activation=CONFIG["activation"],
    )

    # Create agent
    agent = PPOAgent(
        model=model,
        gamma=CONFIG["gamma"],
        lam=CONFIG["lam"],
        pi_lr=CONFIG["pi_lr"],
        vf_lr=CONFIG["vf_lr"],
        n_episodes=CONFIG["n_episodes"],
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        clip_ratio=CONFIG["clip_ratio"],
        target_kl=CONFIG["target_kl"],
    )

    # Create logger
    logger = WandbLogger(
        project=CONFIG["wandb_project"],
        config=CONFIG,
        mode=CONFIG["wandb_mode"],
    )

    # Resume from checkpoint if specified
    start_episode = 0
    total_steps = 0
    if CONFIG["resume_from"]:
        if os.path.exists(CONFIG["resume_from"]):
            start_episode, total_steps = load_checkpoint(CONFIG["resume_from"], model)
            tqdm.write(
                f"Resumed from checkpoint at episode {start_episode}, step {total_steps}"
            )
        else:
            tqdm.write(f"Warning: Checkpoint not found at {CONFIG['resume_from']}")

    tqdm.write(f"Starting training for {CONFIG['total_episodes']} episodes...")
    tqdm.write(f"Observation space: {env.observation_space.shape}")
    tqdm.write(f"Action space: {env.action_space}")
    tqdm.write(f"Self-play: {'enabled' if CONFIG['use_self_play'] else 'disabled'}")

    # Running averages for progress bar display
    recent_rewards: list[float] = []
    recent_win_rate: float | None = None

    # Training loop with progress bar
    pbar = tqdm(
        range(start_episode, CONFIG["total_episodes"]),
        desc="Training",
        unit="ep",
        dynamic_ncols=True,
        initial=start_episode,
        total=CONFIG["total_episodes"],
    )

    for episode in pbar:
        # Train for one episode
        train_stats, ep_steps = train_episode(agent, env)
        total_steps += ep_steps

        # Track recent rewards for smoothed display
        recent_rewards.append(train_stats["train/episode_reward"])
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
        avg_reward = np.mean(recent_rewards)

        # Add step/episode counters to stats
        train_stats["train/episode"] = episode
        train_stats["train/total_steps"] = total_steps

        # Log training stats using total_steps as x-axis for fair comparison
        logger.log_metrics(train_stats, global_step=total_steps)

        # Update progress bar with current stats
        postfix = {
            "reward": f"{avg_reward:.1f}",
            "steps": f"{total_steps:,}",
        }
        if recent_win_rate is not None:
            postfix["win%"] = f"{recent_win_rate:.0%}"
        pbar.set_postfix(postfix)

        # Evaluation
        if (episode + 1) % CONFIG["eval_interval"] == 0:
            eval_stats = evaluate(agent, eval_env, CONFIG["eval_episodes"])
            eval_stats["eval/episode"] = episode
            logger.log_metrics(eval_stats, global_step=total_steps)

            recent_win_rate = eval_stats["eval/win_rate"]
            tqdm.write(
                f"[Eval @ ep {episode + 1}] "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Eval Reward: {eval_stats['eval/mean_reward']:.2f} ± {eval_stats['eval/std_reward']:.2f} | "
                f"Win Rate: {eval_stats['eval/win_rate']:.1%}"
            )

        # Save checkpoint
        if (episode + 1) % CONFIG["save_interval"] == 0:
            ckpt_path = save_checkpoint(
                model=model,
                agent=agent,
                episode=episode + 1,
                total_steps=total_steps,
                checkpoint_dir=checkpoint_dir,
                config=CONFIG,
            )
            tqdm.write(f"[Checkpoint] Saved: {ckpt_path}")

        # Update opponent pool for self-play
        if (
            CONFIG["use_self_play"]
            and opponent_policy is not None
            and episode >= CONFIG["self_play_start_episode"]
            and (episode + 1) % CONFIG["opponent_update_interval"] == 0
        ):
            opponent_policy.add_model(model)
            tqdm.write(
                f"[Self-play] Added model to opponent pool (size: {len(opponent_policy.model_pool)})"
            )

    pbar.close()

    pbar.close()

    # Final evaluation
    tqdm.write("\nRunning final evaluation...")
    final_stats = evaluate(agent, eval_env, CONFIG["eval_episodes"] * 5)
    tqdm.write("\n" + "=" * 50)
    tqdm.write("Final Evaluation:")
    tqdm.write(
        f"  Mean Reward: {final_stats['eval/mean_reward']:.2f} ± {final_stats['eval/std_reward']:.2f}"
    )
    tqdm.write(f"  Win Rate: {final_stats['eval/win_rate']:.1%}")
    tqdm.write("=" * 50)

    # Save final checkpoint
    final_ckpt = save_checkpoint(
        model=model,
        agent=agent,
        episode=CONFIG["total_episodes"],
        total_steps=total_steps,
        checkpoint_dir=checkpoint_dir,
        config=CONFIG,
    )
    tqdm.write(f"Saved final checkpoint: {final_ckpt}")

    logger.close()
    tqdm.write("Training complete!")


if __name__ == "__main__":
    main()
