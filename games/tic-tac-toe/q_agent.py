"""
Tabular Q-learning agent for Tic-Tac-Toe.

The "table" is a Python dict: {state: [Q(s,0), Q(s,1), ..., Q(s,8)]}
We don't need to pre-allocate all ~5,000 reachable states — entries are
created lazily the first time a state is seen.

Update rule (the heart of Q-learning):
    Q(s,a) <- Q(s,a) + alpha * ( reward + gamma * max_a' Q(s',a') - Q(s,a) )

  alpha (learning rate)  : how much each new experience overwrites the old estimate
  gamma (discount factor): how much future reward matters vs immediate reward
  epsilon                : probability of taking a random action instead of the
                            best-known one (exploration vs exploitation)
"""
import random
import pickle
from collections import defaultdict


class QLearningAgent:
    def __init__(self, alpha=0.3, gamma=0.9, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.9995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # Default Q-values start at 0 for every action of a new state
        self.q_table = defaultdict(lambda: [0.0] * 9)

    def choose_action(self, state, available_actions, greedy=False):
        """Epsilon-greedy policy: explore randomly, otherwise exploit best known move."""
        if not greedy and random.random() < self.epsilon:
            return random.choice(available_actions)

        q_values = self.q_table[state]
        # Only consider valid (empty) cells; pick the highest-Q one, breaking ties randomly
        best_value = max(q_values[a] for a in available_actions)
        best_actions = [a for a in available_actions if q_values[a] == best_value]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_available_actions, done):
        current_q = self.q_table[state][action]
        if done or not next_available_actions:
            target = reward
        else:
            target = reward + self.gamma * max(self.q_table[next_state][a] for a in next_available_actions)
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: [0.0] * 9, data)
