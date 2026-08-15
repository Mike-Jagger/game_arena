class QLearningAgent:
    def __init__(self, action_size=3, lr=0.001, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        def _init_(self, action_size=3, lr=0.001, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}
        

    def get_action(self, state, is_training):
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        if is_training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        return int(np.argmax(self.q_table[state_key]))
        
           

    def save(self, filepath):
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        next_key = tuple(next_state) if isinstance(next_state, (list, np.ndarray)) else next_state

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_size)

        best_next_q = 0 if done else np.max(self.q_table[next_key])
        current_q = self.q_table[state_key][action]
        self.q_table[state_key][action] = current_q + self.lr * (reward + self.gamma * best_next_q - current_q)

        if done and self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
    def  save(self, filepath)
         with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
            

    def load(self, filepath):
         with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)