"""
Interactive script to play the Plump card game.

Human player is Player 0, other players can be random agents or trained policies.
"""

import argparse
import glob
import os

import numpy as np
import torch

from rl_temple.environments.plump import PlumpEnv, N_CARDS, N_RANKS
from rl_temple.models.actor_critic import MLPActorCritic
from rl_temple.wrappers.plump_wrapper import PlumpWrapper


def get_valid_actions(action_mask: np.ndarray) -> list[int]:
    """Get list of valid action indices."""
    return list(np.where(action_mask)[0])


def action_to_string(action: int, phase: int) -> str:
    """Convert an action index to a human-readable string."""
    if phase == 0:  # Bidding
        bid = action - N_CARDS
        return f"Bid {bid}"
    else:  # Playing
        suit = action // N_RANKS
        rank = action % N_RANKS
        suit_symbols = ["♣", "♦", "♠", "♥"]
        rank_names = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        return f"{rank_names[rank]}{suit_symbols[suit]}"


def display_hand(hand: np.ndarray) -> None:
    """Display the player's hand in a readable format."""
    suit_symbols = ["♣", "♦", "♠", "♥"]
    rank_names = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

    print("\nYour hand:")
    cards_by_suit: dict[int, list[tuple[int, str]]] = {0: [], 1: [], 2: [], 3: []}

    for card_idx in range(N_CARDS):
        if hand[card_idx]:
            suit = card_idx // N_RANKS
            rank = card_idx % N_RANKS
            cards_by_suit[suit].append((rank, f"{rank_names[rank]}{suit_symbols[suit]}"))

    for suit in range(4):
        if cards_by_suit[suit]:
            # Sort by rank
            cards_by_suit[suit].sort(key=lambda x: x[0])
            cards_str = " ".join(c[1] for c in cards_by_suit[suit])
            print(f"  {suit_symbols[suit]}: {cards_str}")


def display_game_state(obs: dict, env: PlumpEnv) -> None:
    """Display the current game state."""
    suit_names = ["Clubs", "Diamonds", "Spades", "Hearts", "None"]

    print("\n" + "=" * 60)
    print(
        f"Round: {env._round + 1} | Hand Size: {env._hand_size} | Trump: {suit_names[env._trump_suit]}"
    )
    print(f"Phase: {'BIDDING' if obs['phase'] == 0 else 'PLAYING'}")
    print("=" * 60)

    # Show bids and tricks
    print("\nPlayers:")
    for i in range(env.n_players):
        bid_str = str(obs["bids"][i]) if obs["bids"][i] >= 0 else "?"
        player_label = "You" if i == 0 else f"Bot {i}"
        marker = " <--" if i == obs["current_player"] else ""
        print(
            f"  {player_label}: Bid={bid_str}, Tricks={obs['tricks_won'][i]}, Score={obs['scores'][i]}{marker}"
        )

    # Show current trick if playing
    if obs["phase"] == 1 and np.any(obs["current_trick"] >= 0):
        print("\nCurrent Trick:")
        suit_symbols = ["♣", "♦", "♠", "♥"]
        rank_names = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

        for i in range(env.n_players):
            card = obs["current_trick"][i]
            if card >= 0:
                suit = card // N_RANKS
                rank = card % N_RANKS
                card_str = f"{rank_names[rank]}{suit_symbols[suit]}"
            else:
                card_str = "---"
            player_label = "You" if i == 0 else f"Bot {i}"
            print(f"  {player_label}: {card_str}")

        led_suit = obs["led_suit"]
        if led_suit < 4:
            print(f"  Led suit: {suit_names[led_suit]}")

    # Show hand
    display_hand(obs["hand"])


def human_choose_action(valid_actions: list[int], phase: int) -> int:
    """Let the human player choose an action."""
    print("\nValid actions:")
    for i, action in enumerate(valid_actions):
        print(f"  [{i}] {action_to_string(action, phase)}")

    while True:
        try:
            choice = input("\nEnter your choice (number): ").strip()
            idx = int(choice)
            if 0 <= idx < len(valid_actions):
                return valid_actions[idx]
            print(f"Please enter a number between 0 and {len(valid_actions) - 1}")
        except ValueError:
            print("Please enter a valid number")


def random_agent_action(action_mask: np.ndarray, rng: np.random.Generator) -> int:
    """Choose a random valid action for AI players."""
    valid_actions = get_valid_actions(action_mask)
    return rng.choice(valid_actions)


class PolicyAgent:
    """Agent that uses a trained policy model to select actions."""

    def __init__(self, checkpoint_path: str, n_players: int, device: str = "cpu"):
        self.device = device

        # Load checkpoint first to get config
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get("config", {})

        # Get n_players from checkpoint config
        checkpoint_n_players = config.get("n_players", 4)
        if n_players != checkpoint_n_players:
            print(
                f"\n⚠️  Warning: Checkpoint was trained with {checkpoint_n_players} players."
            )
            print(f"    Overriding requested {n_players} players to match checkpoint.")
            n_players = checkpoint_n_players

        self.n_players = n_players

        # Create env and wrapper with correct n_players
        env = PlumpEnv(n_players=n_players)
        wrapper = PlumpWrapper(env, opponent_policy=None, normalize_obs=True)

        # Create model with same architecture
        hidden_sizes = config.get("hidden_sizes", (128, 128))
        activation = config.get("activation", "tanh")

        self.model = MLPActorCritic(
            observation_space=wrapper.observation_space,
            action_space=wrapper.action_space,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.model.to(device)

        # Store wrapper for observation flattening
        self.wrapper = wrapper

        print(f"Loaded policy from: {checkpoint_path}")
        print(f"  Episode: {checkpoint.get('episode', 'unknown')}")
        print(f"  Players: {n_players}")
        print(f"  Architecture: {hidden_sizes}, {activation}")

    def get_action(self, obs_dict: dict, action_mask: np.ndarray) -> int:
        """Select an action using the trained policy."""
        # Flatten observation using wrapper's method
        flat_obs = self.wrapper._flatten_obs(obs_dict)

        with torch.no_grad():
            obs_tensor = torch.as_tensor(
                flat_obs, dtype=torch.float32, device=self.device
            )
            mask_tensor = torch.as_tensor(
                action_mask, dtype=torch.bool, device=self.device
            )

            # Get action from model with action masking
            pi, _ = self.model.pi(
                obs_tensor.unsqueeze(0), action_mask=mask_tensor.unsqueeze(0)
            )
            action = pi.sample().item()

        return int(action)


def list_checkpoints(checkpoint_dir: str = "checkpoints/plump") -> list[str]:
    """List available checkpoint files."""
    if not os.path.exists(checkpoint_dir):
        return []

    checkpoints = []
    # Find all checkpoint files recursively
    for pattern in ["**/checkpoint_*.pt", "**/checkpoint_latest.pt"]:
        checkpoints.extend(
            glob.glob(os.path.join(checkpoint_dir, pattern), recursive=True)
        )

    # Remove duplicates and sort
    checkpoints = sorted(set(checkpoints))
    return checkpoints


def select_checkpoint() -> str | None:
    """Interactive checkpoint selection."""
    checkpoints = list_checkpoints()

    if not checkpoints:
        print("\nNo checkpoints found in checkpoints/plump/")
        print("Train a model first using: python scripts/plump/train.py")
        return None

    print("\nAvailable checkpoints:")
    print("  [0] Random agents (no policy)")
    for i, ckpt in enumerate(checkpoints, 1):
        # Show relative path for readability
        rel_path = os.path.relpath(ckpt)
        print(f"  [{i}] {rel_path}")

    while True:
        try:
            choice = input("\nSelect checkpoint (number): ").strip()
            idx = int(choice)
            if idx == 0:
                return None
            if 1 <= idx <= len(checkpoints):
                return checkpoints[idx - 1]
            print(f"Please enter a number between 0 and {len(checkpoints)}")
        except ValueError:
            print("Please enter a valid number")


def play_game(
    n_players: int = 4,
    seed: int | None = None,
    policy_agent: PolicyAgent | None = None,
) -> None:
    """Play a full game of Plump."""
    env = PlumpEnv(n_players=n_players, render_mode=None)
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    print("\n" + "=" * 60)
    print("    WELCOME TO PLUMP!")
    print("=" * 60)
    print(f"\nPlaying with {n_players} players. You are Player 0.")
    print("Goal: Bid exactly how many tricks you'll win each round!")
    print("Scoring: Hit bid = 10 + bid points (or 5 for bid 0)")
    print("         Miss bid = 0 points (a 'plump')")

    terminated = False
    total_reward = 0.0

    while not terminated:
        current_player = obs["current_player"]
        action_mask = info["action_mask"]
        valid_actions = get_valid_actions(action_mask)

        if current_player == 0:
            # Human player's turn
            display_game_state(obs, env)
            action = human_choose_action(valid_actions, obs["phase"])
        else:
            # AI player's turn - use policy if available, otherwise random
            if policy_agent is not None:
                action = policy_agent.get_action(obs, action_mask)
                bot_type = "AI"
            else:
                action = random_agent_action(action_mask, rng)
                bot_type = "Bot"

            # Show what the AI did (brief)
            phase = obs["phase"]
            if phase == 0:
                print(f"\n{bot_type} {current_player} bids {action - N_CARDS}")
            else:
                suit_symbols = ["♣", "♦", "♠", "♥"]
                rank_names = [
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                    "10",
                    "J",
                    "Q",
                    "K",
                    "A",
                ]
                suit = action // N_RANKS
                rank = action % N_RANKS
                print(
                    f"\n{bot_type} {current_player} plays {rank_names[rank]}{suit_symbols[suit]}"
                )

        obs, reward, terminated, truncated, info = env.step(action)

        if reward != 0:
            total_reward += reward
            print(f"\n*** Round ended! You earned {reward} points this round. ***")
            print(f"*** Total score: {obs['scores'][0]} ***")
            if not terminated:
                input("\nPress Enter to continue to the next round...")

    # Game over
    print("\n" + "=" * 60)
    print("    GAME OVER!")
    print("=" * 60)
    print("\nFinal Scores:")
    for i in range(n_players):
        player_label = "You" if i == 0 else f"Bot {i}"
        print(f"  {player_label}: {obs['scores'][i]}")

    winner = np.argmax(obs["scores"])
    if winner == 0:
        print("\n🎉 Congratulations! You won! 🎉")
    else:
        bot_type = "AI" if policy_agent else "Bot"
        print(f"\n{bot_type} {winner} wins!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Play Plump against AI or random bots")
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=None,
        help="Path to a checkpoint file to use as opponent policy",
    )
    parser.add_argument(
        "--players", "-p", type=int, default=None, help="Number of players (2-5)"
    )
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    # Get checkpoint first (so we can read n_players from it)
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = select_checkpoint()

    # Get number of players (may be overridden by checkpoint)
    if args.players is not None:
        n_players = args.players
    elif checkpoint_path is not None:
        # Read n_players from checkpoint config
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        n_players = config.get("n_players", 4)
        print(f"\nUsing {n_players} players (from checkpoint config)")
    else:
        print("\nHow many players? (2-5)")
        while True:
            try:
                n_players = int(input("Enter number of players: ").strip())
                if 2 <= n_players <= 5:
                    break
                print("Please enter a number between 2 and 5")
            except ValueError:
                print("Please enter a valid number")

    # Load policy agent if checkpoint specified
    policy_agent = None
    if checkpoint_path is not None:
        policy_agent = PolicyAgent(checkpoint_path, n_players)
        # Update n_players in case checkpoint overrode it
        n_players = policy_agent.n_players

    # Get seed
    if args.seed is not None:
        seed = args.seed
    else:
        seed_input = input("Enter random seed (or press Enter for random): ").strip()
        seed = int(seed_input) if seed_input else None

    play_game(n_players=n_players, seed=seed, policy_agent=policy_agent)

    while True:
        again = input("\nPlay again? (y/n): ").strip().lower()
        if again == "y":
            play_game(n_players=n_players, seed=None, policy_agent=policy_agent)
        elif again == "n":
            print("Thanks for playing!")
            break


if __name__ == "__main__":
    main()
