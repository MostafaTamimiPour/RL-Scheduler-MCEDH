"""
agent.py - Single Agent with Q-Learning, SARSA, and SARSA(λ)

HYBRID RULE + RL DESIGN
------------------------
The agent never picks from the full action list. Before every choice it asks
the environment for ``valid_actions`` via ``env.get_valid_actions()``. That
list already has every unsafe or HI-mandatory action removed (see the
docstring at the top of environment.py: Rules 3/4 and the LO-in-HI safety
filter are enforced there, deterministically, before RL is ever consulted).

As a result:
  * epsilon-greedy exploration can only ever explore *safe* actions,
  * the greedy/argmax step can only ever prefer *safe* actions,
  * Q-Learning's bootstrap (max_a' Q(s', a')) only maximizes over actions
    that will actually be legal in the next state.

RL therefore only ever learns inside the genuine discretion zone (ED-H's
Rule 5): which safe task to run, or whether to idle to bank energy. It
cannot learn its way into an unsafe HI-endangering choice because that
choice is never presented as an option.
"""

import random
import numpy as np


class RLAgent:
    def __init__(self, env, algorithm='q_learning', alpha=0.5, gamma=0.9, lam=0.9,
                 epsilon_start=1.0, epsilon_min=0.005, decay_type='exponential'):
        """
        algorithm: 'q_learning', 'sarsa', 'sarsa_lambda'
        """
        self.env = env
        self.algorithm = algorithm
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam  # only used by SARSA(λ)
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.decay_type = decay_type

        self.Q = {}
        self.E = {}  # only used by SARSA(λ)
        self.rewards_history = []
        self.eps_history = []
        # Diagnostics: how often the agent actually had a real choice to
        # make (Rule 5) vs. how often Rules 3/4 forced the action for it.
        self.forced_step_count = 0
        self.free_step_count = 0

    def get_epsilon(self, episode, num_episodes):
        if self.decay_type == 'linear':
            return max(self.epsilon_min, self.epsilon_start * (1 - episode / num_episodes))
        elif self.decay_type == 'sqrt':
            return max(self.epsilon_min, self.epsilon_start / np.sqrt(episode + 1))
        else:  # exponential
            return max(self.epsilon_min, self.epsilon_start * (0.999 ** episode))

    def _ensure_state(self, state):
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in range(len(self.env.actions))}
            if self.algorithm == 'sarsa_lambda':
                self.E[state] = {a: 0.0 for a in range(len(self.env.actions))}

    def get_action(self, state, epsilon, valid_actions):
        """Epsilon-greedy selection, restricted to ``valid_actions``.

        ``valid_actions`` must come from ``env.get_valid_actions()`` for the
        state being acted on. This is what keeps RL confined to the Rule-5
        discretion zone: it is never even asked to consider an unsafe or
        HI-mandatory action, so it cannot explore into one.
        """
        self._ensure_state(state)

        if not valid_actions:
            raise ValueError("valid_actions must be non-empty")

        if random.random() < epsilon:
            return random.choice(valid_actions)

        q_vals = {a: self.Q[state][a] for a in valid_actions}
        best_value = max(q_vals.values())
        best = [a for a, v in q_vals.items() if v == best_value]
        return random.choice(best)

    def reset_traces(self):
        """Reset eligibility traces (only for SARSA(λ))"""
        if self.algorithm == 'sarsa_lambda':
            for state in self.Q:
                self.E[state] = {a: 0.0 for a in range(len(self.env.actions))}

    def update(self, state, action, reward, next_state, next_action, done, next_valid_actions):
        """Update Q-value based on selected algorithm.

        ``next_valid_actions`` restricts Q-Learning's max-bootstrap to
        actions that will actually be legal in ``next_state`` -- otherwise
        the agent could bootstrap off the value of an action it could never
        actually take, inflating estimates for states near a forced switch.
        """
        self._ensure_state(state)
        self._ensure_state(next_state)

        # ===== TD error =====
        if self.algorithm == 'q_learning':
            if done:
                next_q = 0.0
            else:
                next_q = max(self.Q[next_state][a] for a in next_valid_actions)
        else:
            # SARSA / SARSA(λ): bootstrap off the actual next action taken
            next_q = self.Q[next_state][next_action] if not done else 0.0

        td_error = reward + self.gamma * next_q - self.Q[state][action]

        # ===== Update =====
        if self.algorithm in ('q_learning', 'sarsa'):
            self.Q[state][action] += self.alpha * td_error

        elif self.algorithm == 'sarsa_lambda':
            # 1. Decay all traces
            for s in self.Q:
                for a in range(len(self.env.actions)):
                    self.E[s][a] = self.gamma * self.lam * self.E[s][a]
            # 2. Add 1 to current (state, action)
            self.E[state][action] += 1.0
            # 3. Update Q for all (state, action)
            for s in self.Q:
                for a in range(len(self.env.actions)):
                    self.Q[s][a] += self.alpha * td_error * self.E[s][a]

    def train(self, episodes=3000):
        self.rewards_history = []
        self.eps_history = []
        self.forced_step_count = 0
        self.free_step_count = 0

        for ep in range(episodes):
            epsilon = self.get_epsilon(ep, episodes)
            self.eps_history.append(epsilon)

            state = self.env.reset()
            valid_actions, forced = self.env.get_valid_actions()

            if self.algorithm == 'q_learning':
                action = None  # chosen inside the loop, once masking is known
            else:
                action = self.get_action(state, epsilon, valid_actions)
                self.reset_traces()  # only for SARSA(λ)

            done = False
            total_reward = 0
            steps = 0

            while not done and steps < 50:
                if forced:
                    self.forced_step_count += 1
                else:
                    self.free_step_count += 1

                if self.algorithm == 'q_learning':
                    action = self.get_action(state, epsilon, valid_actions)

                next_state, reward, done = self.env.step(action)
                total_reward += reward
                steps += 1

                if done:
                    next_valid_actions, next_forced = valid_actions, forced
                else:
                    next_valid_actions, next_forced = self.env.get_valid_actions()
                next_action = self.get_action(next_state, epsilon, next_valid_actions)

                self.update(state, action, reward, next_state, next_action, done, next_valid_actions)

                state = next_state
                valid_actions, forced = next_valid_actions, next_forced
                if self.algorithm != 'q_learning':
                    action = next_action

            self.rewards_history.append(total_reward)
            if ep % 100 == 0:
                total_choice_steps = self.forced_step_count + self.free_step_count
                free_pct = (100 * self.free_step_count / total_choice_steps) if total_choice_steps else 0.0
                print(f"{self.algorithm.upper()} Episode {ep}: steps={steps}, reward={total_reward}, "
                      f"epsilon={epsilon:.3f}, RL-free-choice-steps-so-far={free_pct:.1f}%")

        return self.Q, self.rewards_history, self.eps_history
