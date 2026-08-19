import argparse
import os
import json
import neat
import pickle
import numpy as np
from games.snake import SnakeGame
from games.tictactoe import TicTacToeGame
from games.flappy_bird import FlappyBirdGame
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

def train_snake_neat(generations=500):
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
    # print("Winner details:", winner)
    metrics = {
        "episodes": generations,
        "final_avg_score": getattr(winner, 'fitness', 0), 
        "max_score": getattr(winner, 'fitness', 0)
    }
    with open(os.path.join(STORAGE_DIR, "snake_neat_metrics.json"), "w") as f:
        json.dump(metrics, f)
        
    print(f"NEAT Training complete. Best genome saved to {winner_path}.")

def train_flappy_qlearning(episodes=2000):
    env = FlappyBirdGame()
    agent = QLearningAgent(action_size=2)
    scores = []
    
    for ep in range(episodes):
        env.reset()
        state = env.get_state(discrete=True)
        done = False
        while not done:
            action = agent.get_action(state, is_training=True)
            _, reward, done, score = env.step(action)
            next_state = env.get_state(discrete=True)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        scores.append(score)
        if ep % 100 == 0:
            print(f"Episode {ep}/{episodes} | Avg Score: {sum(scores[-100:])/max(1, len(scores[-100:])):.2f}")

    agent.save(os.path.join(STORAGE_DIR, "flappybird_qlearning.pkl"))
    metrics = {"final_avg_score": sum(scores[-100:]) / 100, "max_score": max(scores), "scores_history": scores}
    with open(os.path.join(STORAGE_DIR, "flappybird_qlearning_metrics.json"), 'w') as f:
        json.dump(metrics, f)
    print("Flappy Q-Learning Training finished.")

def train_flappy_dqn(episodes=1000):
    env = FlappyBirdGame()
    agent = DQNAgent(input_dim=4, output_dim=2)
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

    agent.save(os.path.join(STORAGE_DIR, "flappybird_dqn.pt"))
    metrics = {"final_avg_score": sum(scores[-50:]) / 50, "max_score": max(scores), "scores_history": scores, "loss_history": losses}
    with open(os.path.join(STORAGE_DIR, "flappybird_dqn_metrics.json"), 'w') as f:
        json.dump(metrics, f)
    print("Flappy DQN Training complete.")

def eval_flappy_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        env = FlappyBirdGame()
        state = env.reset()
        done = False
        
        while not done:
            output = net.activate(state)
            action = np.argmax(output)
            state, reward, done, score = env.step(action)
            genome.fitness += reward

def train_flappy_neat(generations=500):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, "config/neat_flappy.cfg")
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    winner = p.run(eval_flappy_genomes, generations)
    
    with open(os.path.join(STORAGE_DIR, "flappybird_neat.pkl"), "wb") as f:
        pickle.dump(winner, f)
        
    metrics = {"final_avg_score": getattr(winner, 'fitness', 0), "max_score": getattr(winner, 'fitness', 0)}
    with open(os.path.join(STORAGE_DIR, "flappybird_neat_metrics.json"), "w") as f:
        json.dump(metrics, f)
    print("Flappy NEAT Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ML algorithms on Snake or TicTacToe")
    parser.add_argument('--game', choices=['snake', 'flappybird'], required=True)
    parser.add_argument('--model', choices=['qlearning', 'dqn', 'neat'], required=True)
    parser.add_argument('--episodes', type=int, default=0)
    parser.add_argument('--all', choices=['games', 'snake', 'flappybird'], default=['games'])
    args = parser.parse_args()

    if args.game == 'snake' or args.all in ['games', 'snake']:
        if args.model == 'qlearning':
            train_snake_qlearning(args.episodes) if args.episodes > 0 else train_snake_qlearning()
        elif args.model == 'dqn':
            train_snake_dqn(args.episodes) if args.episodes > 0 else train_snake_dqn()
        elif args.model == 'neat':
            train_snake_neat(args.episodes) if args.episodes > 0 else train_snake_neat()
    elif args.game == 'flappybird' or args.all in ['games', 'flappybird']:
        if args.model == 'qlearning':
            train_flappy_qlearning(args.episodes) if args.episodes > 0 else train_flappy_qlearning()
        elif args.model == 'dqn':
            train_flappy_dqn(args.episodes) if args.episodes > 0 else train_flappy_dqn()
        elif args.model == 'neat':
            train_flappy_neat(args.episodes) if args.episodes > 0 else train_flappy_neat()