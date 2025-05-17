import numpy as np

Transition = tuple[np.ndarray, np.ndarray, float, float, float, bool]


class RolloutBuffer:

    def __init__(self) -> None:
        self.clear()

    def add(self, transition: Transition) -> None:
        state, action, reward, log_prob, value, done = transition

        self.current_states.append(state)
        self.current_actions.append(action)
        self.current_rewards.append(reward)
        self.current_log_probs.append(log_prob)
        self.current_values.append(value)
        self.current_dones.append(done)

    def end_episode(self) -> None:
        episode = {
            "states": np.array(self.current_states),
            "actions": np.array(self.current_actions),
            "rewards": np.array(self.current_rewards),
            "log_probs": np.array(self.current_log_probs),
            "values": np.array(self.current_values),
            "dones": np.array(self.current_dones),
        }
        self.trajectories.append(episode)

        self.current_states = []
        self.current_actions = []
        self.current_rewards = []
        self.current_log_probs = []
        self.current_values = []
        self.current_dones = []

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

    def __len__(self) -> int:
        return len(self.trajectories)
