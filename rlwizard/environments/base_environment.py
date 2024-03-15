class BaseEnvironment:
    def __init__(self) -> None:
        self.observation_space = None
        self.action_space = None

    def reset(self):
        """
        Reset the environment and return the initial observation.
        """
        raise NotImplementedError("reset method must be implemented")

    def step(self, action):
        """
        Take a step in the environment based on the given action.
        Return the next observation, reward, done flag and additional_info.
        """
        raise NotImplementedError("step method must be implemented")
