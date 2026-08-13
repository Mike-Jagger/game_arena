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