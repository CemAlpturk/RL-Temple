from gymnasium.envs.registration import register

register(
    id="temple/Snake-v0",
    entry_point="temple.environments:SnakeEnv",
    max_episode_steps=1000,
)
