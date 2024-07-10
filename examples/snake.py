import numpy as np
from gymnasium import Env, ObservationWrapper, spaces, RewardWrapper

import torch.nn as nn

from temple.algos.dqn import DQNAgent
from temple.environments.snake_env import SnakeEnv


class AddChannelToObs(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=4,
            shape=(1, *shape),
            dtype=np.int32,
        )

    def observation(self, observation):
        return observation[None, :]


class RewardShaping(RewardWrapper):
    def __init__(self, env: Env):
        super().__init__(env)

    def reward(self, reward):
        return reward


def env_fn(render_mode=None):
    return AddChannelToObs(SnakeEnv(render_mode=render_mode))


def main():
    env = env_fn()
    # model_config = {
    #     "type": "mlp",
    #     "args": {
    #         "input_size": env.observation_space.shape[0],
    #         "output_size": env.action_space.n,
    #         "hidden_sizes": [64, 64],
    #         "activation": "relu",
    #         "final_activation": None,
    #     },
    # }

    model = nn.Sequential(
        nn.Conv2d(1, 64, 4),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, 2),
        nn.ReLU(),
        nn.MaxPool2d(2, 1),
        nn.Flatten(),
        # nn.Linear(64 * 2 * 2, 256),
        # nn.ReLU(),
        nn.Linear(64, env.action_space.n),
    )
    # model = nn.Sequential(
    #     nn.Linear(env.observation_space.shape[0], 64),
    #     nn.ReLU(),
    #     nn.Linear(64, 64),
    #     nn.ReLU(),
    #     nn.Linear(64, env.action_space.n),
    # )

    agent = DQNAgent(
        env_fn=env_fn,
        # model_config=model_config,
        model=model,
        learning_rate=0.001,
        gamma=0.97,
        epsilon=1.0,
        epsilon_delta=1e-5,
        epsilon_min=0.01,
        batch_size=32,
        memory_size=50000,
        target_update=1000,
        max_steps_per_episode=1000,
        n_episodes=20000,
        n_eval_episodes=10,
        eval_interval=100,
        train_per_step=1,
        record_env_interval=10000,
    )

    agent.train()

    env = env_fn("human")
    while True:
        state, _ = env.reset()
        step = 0
        done = False
        while not done and step < 100:
            action = agent.choose_action(state, training=False)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            env.render()
            step += 1


if __name__ == "__main__":
    main()
