import neat
import pickle
import numpy as np

class NEATAgent:
    def __init__(self, config_path):
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        self.net = None

    def get_action(self, state):
        output = self.net.activate(state)
        return int(np.argmax(output))

    def save(self, winner_genome, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(winner_genome, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            winner = pickle.load(f)
        self.net = neat.nn.FeedForwardNetwork.create(winner, self.config)
