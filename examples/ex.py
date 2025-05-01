import gymnasium as gym
import torch
from rl_temple.trainer import Trainer
from rl_temple.agents.dqn_agent import DQNAgent
from rl_temple.runners.off_policy import OffPolicyRunner


env_fn = lambda: gym.make("CartPole-v1")
env = env_fn()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model_config = {
    "type": "mlp",
    "args": {
        "input_size": state_dim,
        "output_size": action_dim,
        "hidden_sizes": [64, 64, 64],
    },
}

agent = DQNAgent(
    action_dim=action_dim,
    model_config=model_config,
    device=torch.device("cuda"),
)
runner = OffPolicyRunner(agent, env)
trainer = Trainer(runner, env_fn, num_episodes=1000, eval_interval=20)
trainer.train()

env = gym.make("CartPole-v1", render_mode="human")
while True:

    state, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = agent.select_action(state, explore=False)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)

    print(f"Total Reward: {total_reward}")
