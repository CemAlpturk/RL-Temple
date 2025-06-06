from rl_temple.runners.base import AgentRunner


class OffPolicyRunner(AgentRunner):
    """For agents that update every step."""

    def run_episode(
        self, max_steps: int | None, explore: bool = True
    ) -> dict[str, float]:
        state, _ = self.env.reset()
        total_reward = 0.0
        step = 0
        while True:
            action = self.agent.select_action(state, explore=explore)
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            self.agent.remember(state, action, reward, next_state, terminated)
            self.agent.update()

            state = next_state
            total_reward += float(reward)
            step += 1

            if max_steps is not None and step >= max_steps:
                break
            if terminated or truncated:
                break

        stats = self.agent.end_episode()
        return {
            "total_reward": total_reward,
            "steps": step,
            **stats,
        }
