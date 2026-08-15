class DQNAgent:
  class LinearQNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearQNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)
    
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
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

     def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        mini_sample = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        states_t = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool).to(self.device)

        pred = self.model(states_t).gather(1, actions_t).squeeze()
        next_q = self.model(next_states_t).max(1)[0]
        next_q[dones_t] = 0.0
        target = rewards_t + self.gamma * next_q

        loss = self.criterion(pred, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        return loss.item()
   

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)

        

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
      