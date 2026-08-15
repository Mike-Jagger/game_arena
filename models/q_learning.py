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
        pass

    def load(self, filepath):
        pass