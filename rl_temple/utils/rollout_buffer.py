class RolloutBuffer:

    def __init__(self) -> None:
        self.clear()

    def add(self, transition: tuple) -> None:
        self.states.append(transition[0])
        self.actions.append(transition[1])
        self.rewards.append(transition[2])
        self.log_probs.append(transition[3])
        self.values.append(transition[4])
        self.dones.append(transition[5])

    def get(self) -> tuple:
        return (
            self.states,
            self.actions,
            self.rewards,
            self.log_probs,
            self.values,
            self.dones,
        )

    def clear(self) -> None:
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
