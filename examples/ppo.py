import gymnasium as gym
import ale_py
import torch

from rl_temple.trainer import Trainer
from rl_temple.agents.ppo_agent import PPOAgent
from rl_temple.runners import OnPolicyRunner

gym.register_envs(ale_py)


def env_fn():
    return gym.make("LunarLander-v3")


env = env_fn()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model_config = {
    "actor": {
        "type": "mlp",
        "args": {
            "input_size": state_dim,
            "output_size": action_dim,
            "hidden_sizes": [128, 128],
            "activation": "tanh",
        },
    },
    "critic": {
        "type": "mlp",
        "args": {
            "input_size": state_dim,
            "output_size": 1,
            "hidden_sizes": [128, 128],
            "activation": "tanh",
        },
    },
}

agent = PPOAgent(
    action_dim=action_dim,
    model_config=model_config,
    device=torch.device("cuda"),
)
runner = OnPolicyRunner(agent, env)
trainer = Trainer(
    runner,
    env_fn,
    num_episodes=10000,
    eval_interval=20,
    eval_env_fn=env_fn,
)
trainer.train()

env = gym.make("LunarLander-v3", render_mode="human")
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
