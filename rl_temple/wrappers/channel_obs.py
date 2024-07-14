from gymnasium import spaces, ObservationWrapper


class ChannelAddedObservation(ObservationWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=env.observation_space.low[None, :],
            high=env.observation_space.high[None, :],
            shape=(1, *shape),
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation):
        return observation[None, :]
