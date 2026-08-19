import argparse
import os
import json
import neat
import pickle
from games.snake import SnakeGame
from games.tictactoe import TicTacToeGame
from models.q_learning import QLearningAgent
from models.dqn import DQNAgent
from models.neat_agent import NEATAgent

STORAGE_DIR = "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

def train_snake_qlearning(episodes=1000):
    env = SnakeGame()
    agent = QLearningAgent(action_size=3)
    scores = []
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state, is_training=True)
            next_state, reward, done, score = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        scores.append(score)
        if ep % 100 == 0:
            print(f"Episode {ep}/{episodes} | Avg Score: {sum(scores[-100:])/max(1, len(scores[-100:])):.2f}")

    # Persist agent & metrics
    agent.save(os.path.join(STORAGE_DIR, "snake_qlearning.pkl"))
    metrics = {
        "episodes": episodes,
        "final_avg_score": sum(scores[-100:]) / 100,
        "max_score": max(scores),
        "scores_history": scores
    }
    with open(os.path.join(STORAGE_DIR, "snake_qlearning_metrics.json"), 'w') as f:
        json.dump(metrics, f)
    print("Training finished & artifacts saved.")

def train_snake_dqn(episodes=500):
    env = SnakeGame()
    agent = DQNAgent(input_dim=11, output_dim=3)
    scores, losses = [], []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_loss = 0
        steps = 0
        while not done:
            action = agent.get_action(state, is_training=True)
            next_state, reward, done, score = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step()
            total_loss += loss
            steps += 1
            state = next_state
        scores.append(score)
        losses.append(total_loss / max(1, steps))
        if ep % 50 == 0:
            print(f"Episode {ep}/{episodes} | Avg Score: {sum(scores[-50:])/max(1, len(scores[-50:])):.2f}")

    agent.save(os.path.join(STORAGE_DIR, "snake_dqn.pt"))
    metrics = {
        "episodes": episodes,
        "final_avg_score": sum(scores[-50:]) / 50,
        "max_score": max(scores),
        "scores_history": scores,
        "loss_history": losses
    }
    with open(os.path.join(STORAGE_DIR, "snake_dqn_metrics.json"), 'w') as f:
        json.dump(metrics, f)
    print("DQN Training complete.")

def eval_genomes(genomes, config):
    """
    Evaluates the fitness of a population of genomes in the Snake environment.
    Fitness is heavily weighted towards surviving longer and eating food.
    """
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        env = SnakeGame()
        state = env.reset()
        done = False
        
        while not done:
            # Get output from the neural network and pick the action with the highest value
            output = net.activate(state)
            action = np.argmax(output)
            
            next_state, reward, done, score = env.step(action)
            
            # Custom fitness calculation for the evolutionary algorithm
            genome.fitness += reward
            state = next_state

def train_snake_neat(generations=50):
    config_path = "config/neat_snake.cfg"
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path
    )
    
    # Initialize Population
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # Run Evolution
    winner = p.run(eval_genomes, generations)
    
    # Persist the winning genome
    winner_path = os.path.join(STORAGE_DIR, "snake_neat.pkl")
    with open(winner_path, "wb") as f:
        pickle.dump(winner, f)
        
    # Persist metrics for visualizations
    metrics = {
        "episodes": generations,
        "final_avg_score": getattr(winner, 'fitness', 0), 
        "max_score": getattr(winner, 'fitness', 0)
    }
    with open(os.path.join(STORAGE_DIR, "snake_neat_metrics.json"), "w") as f:
        json.dump(metrics, f)
        
    print(f"NEAT Training complete. Best genome saved to {winner_path}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ML algorithms on Snake or TicTacToe")
    parser.add_argument('--game', choices=['snake', 'tictactoe'], required=True)
    parser.add_argument('--model', choices=['qlearning', 'dqn', 'neat'], required=True)
    parser.add_argument('--episodes', type=int, default=500)
    args = parser.parse_args()

    if args.game == 'snake':
        if args.model == 'qlearning':
            train_snake_qlearning(args.episodes)
        elif args.model == 'dqn':
            train_snake_dqn(args.episodes)