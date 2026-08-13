import pygame
import os
import json
import sys
from games.snake import SnakeGame
from models.q_learning import QLearningAgent
from models.dqn import DQNAgent
from models.neat_agent import NEATAgent

VIEW_WIDTH = 300
VIEW_HEIGHT = 300
PANEL_HEIGHT = 150
SCREEN_WIDTH = VIEW_WIDTH * 3 + 40
SCREEN_HEIGHT = VIEW_HEIGHT + PANEL_HEIGHT

def load_metrics(game, model_name):
    filepath = f"storage/{game}_{model_name}_metrics.json"
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {"final_avg_score": "N/A", "max_score": "N/A"}
def run_arena(game_type="snake"):
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Game Arena - Simultaneous Model Comparison")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)
    header_font = pygame.font.SysFont("monospace", 16, bold=True)

    # Instantiate separate game environments
    env_ql = SnakeGame(VIEW_WIDTH, VIEW_HEIGHT)
    env_dqn = SnakeGame(VIEW_WIDTH, VIEW_HEIGHT)
    env_neat = SnakeGame(VIEW_WIDTH, VIEW_HEIGHT)
        # Load agents
    agent_ql = QLearningAgent(action_size=3)
    agent_ql.load("storage/snake_qlearning.pkl")
    m_ql = load_metrics("snake", "qlearning")

    agent_dqn = DQNAgent(input_dim=11, output_dim=3)
    agent_dqn.load("storage/snake_dqn.pt")
    m_dqn = load_metrics("snake", "dqn")
        # Tracking states
    states = [env_ql.reset(), env_dqn.reset(), env_neat.reset()]
    dones = [False, False, False]
    scores = [0, 0, 0]
    steps = [0, 0, 0]

    surfaces = [
        pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT)),
        pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT)),
        pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
    ]

    running = True
        while running:
        clock.tick(15)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Step Q-Learning
        if not dones[0]:
            action = agent_ql.get_action(states[0], is_training=False)
            states[0], _, dones[0], scores[0] = env_ql.step(action)
            steps[0] += 1

        # Step DQN
        if not dones[1]:
            action = agent_dqn.get_action(states[1], is_training=False)
            states[1], _, dones[1], scores[1] = env_dqn.step(action)
            steps[1] += 1