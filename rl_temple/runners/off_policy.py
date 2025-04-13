from rl_temple.runners.base import AgentRunner


class OffPolicyRunner(AgentRunner):
    """For agents that update every step."""

    def run_episode(self, max_steps: int, explore: bool = True) -> float:
        state, _ = self.env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action = self.agent.select_action(state, explore=explore)
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            self.agent.remember(state, action, reward, next_state, terminated)
            self.agent.update()

            state = next_state
            total_reward += float(reward)

            if terminated or truncated:
                break

        self.agent.end_episode()
        return total_reward
