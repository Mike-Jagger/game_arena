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
      