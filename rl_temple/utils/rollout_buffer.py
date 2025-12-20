from typing import Callable
import numpy as np

Transition = tuple[np.ndarray, np.ndarray, float, float, float, bool]
TransitionWithMask = tuple[
    np.ndarray, np.ndarray, float, float, float, bool, np.ndarray | None
]


class RolloutBuffer:

    def __init__(
        self,
        postprocess_fn: (
            Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None
        ) = None,
    ) -> None:
        self.clear()
        self.postprocess_fn = postprocess_fn

    def add(self, transition: Transition | TransitionWithMask) -> None:
        if len(transition) == 6:
            state, action, reward, log_prob, value, done = transition
            action_mask = None
        else:
            state, action, reward, log_prob, value, done, action_mask = transition

        self.current_states.append(state)
        self.current_actions.append(action)
        self.current_rewards.append(reward)
        self.current_log_probs.append(log_prob)
        self.current_values.append(value)
        self.current_dones.append(done)
        if action_mask is not None:
            self.current_action_masks.append(action_mask)

    def end_episode(self) -> None:
        episode = {
            "states": np.array(self.current_states),
            "actions": np.array(self.current_actions),
            "rewards": np.array(self.current_rewards),
            "log_probs": np.array(self.current_log_probs),
            "values": np.array(self.current_values),
            "dones": np.array(self.current_dones),
        }
        if self.current_action_masks:
            episode["action_masks"] = np.array(self.current_action_masks)

        episode = self.postprocess_fn(episode) if self.postprocess_fn else episode
        self.trajectories.append(episode)

        self.current_states = []
        self.current_actions = []
        self.current_rewards = []
        self.current_log_probs = []
        self.current_values = []
        self.current_dones = []
        self.current_action_masks = []

    def get(self) -> list[dict[str, np.ndarray]]:
        return self.trajectories

    def clear(self) -> None:
        self.trajectories: list[dict[str, np.ndarray]] = []

        self.current_states: list[np.ndarray] = []
        self.current_actions: list[np.ndarray] = []
        self.current_rewards: list[float] = []
        self.current_log_probs: list[float] = []
        self.current_values: list[float] = []
        self.current_dones: list[bool] = []
        self.current_action_masks: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.trajectories)
