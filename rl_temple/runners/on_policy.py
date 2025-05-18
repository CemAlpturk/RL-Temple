from rl_temple.runners.base import AgentRunner


class OnPolicyRunner(AgentRunner):
    """For agents that collect rollout, then update."""

    def run_episode(
        self, max_steps: int | None, explore: bool = True
    ) -> dict[str, float]:
        state, _ = self.env.reset()
        total_reward = 0.0

        step = 0
        while True:
            action, info = self.agent.select_action(
                state, explore=explore, return_info=True
            )
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            self.agent.remember(
                state,
                action,
                reward,
                next_state,
                terminated,
                log_prob=info[0],
                value=info[1],
            )

            total_reward += float(reward)
            state = next_state

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
