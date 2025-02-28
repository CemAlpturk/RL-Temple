import gymnasium as gym

import torch.nn as nn

from rl_temple.algos.dqn import DQNAgent
from rl_temple.wrappers import ChannelAddedObservation


def main():
    env = gym.make("rl_temple/Snake-v0")
    env = ChannelAddedObservation(env)

    test_env = gym.make("rl_temple/Snake-v0", render_mode="rgb_array_list")
    test_env = ChannelAddedObservation(test_env)

    model = nn.Sequential(
        nn.Conv2d(1, 64, 4),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, 2),
        nn.ReLU(),
        nn.MaxPool2d(2, 1),
        nn.Flatten(),
        nn.Linear(64, env.action_space.n),
    )

    agent = DQNAgent(
        env=env,
        model=model,
        test_env=test_env,
        learning_rate=0.001,
        gamma=0.97,
        epsilon=1.0,
        epsilon_delta=5e-6,
        epsilon_min=0.01,
        batch_size=32,
        memory_size=50000,
        target_update=1000,
        max_steps=int(10e6),
        n_eval_episodes=20,
        eval_interval=10000,
        train_per_step=1,
        record_env_interval=50000,
    )

    agent.train()

    env = gym.make("rl_temple/Snake-v0", render_mode="human")
    env = ChannelAddedObservation(env)

    while True:
        state, _ = env.reset()
        step = 0
        done = False
        while not done:
            action = agent.choose_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            step += 1


if __name__ == "__main__":
    main()
