from gymnasium.envs.registration import register

register(
    id="rl_temple/Snake-v0",
    entry_point="rl_temple.environments:SnakeEnv",
    max_episode_steps=1000,
)

register(
    id="TicTacToe-v0",
    entry_point="rl_temple.environments:TicTacToeEnv",
)
