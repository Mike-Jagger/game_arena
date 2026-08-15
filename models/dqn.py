class DQNAgent:
    def __init__(self, input_dim=11, output_dim=3, lr=0.001, gamma=0.9):
        self.gamma = gamma
        self.output_dim = output_dim
        self.memory = deque(maxlen=20000)
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LinearQNet(input_dim, 128, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
      

    def get_action(self, state, is_training):
        if is_training and random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1)
        state_t = torch.tensor(state, dtype=torch.float).to(self.device)
        with torch.no_grad():
            prediction = self.model(state_t)
        return int(torch.argmax(prediction).item())
    
        

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
      