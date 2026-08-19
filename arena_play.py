import pygame
import os
import json
import sys
from games.snake import SnakeGame
from models.q_learning import QLearningAgent
from models.dqn import DQNAgent
from models.neat_agent import NEATAgent

# Always make sure the width//10 is an even number, because the snake moves in increments of 20 pixels (BLOCK_SIZE). 
# An odd width can cause the snake to go out of bounds.
VIEW_WIDTH = 400
VIEW_HEIGHT = 400
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

    try:
        agent_ql.load("storage/snake_qlearning.pkl")
        m_ql = load_metrics("snake", "qlearning")
    except FileNotFoundError:
        print("Q-Learning model not found. Run 'python train.py --game snake --model qlearning' first.")
        sys.exit(1)

    agent_dqn = DQNAgent(input_dim=11, output_dim=3)
    try:
        agent_dqn.load("storage/snake_dqn.pt")
        m_dqn = load_metrics("snake", "dqn")
    except FileNotFoundError:
        print("DQN model not found. Run 'python train.py --game snake --model dqn' first.")
        sys.exit(1)

    agent_neat = NEATAgent("config/neat_snake.cfg")
    try:
        agent_neat.load("storage/snake_neat.pkl")
        m_neat = load_metrics("snake", "neat")
    except FileNotFoundError:
        print("NEAT model not found. Run 'python train.py --game snake --model neat' first.")
        sys.exit(1)

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

        # step NEAT
        if not dones[2]:
            action = agent_neat.get_action(states[2])
            states[2], _, dones[2], scores[2] = env_neat.step(action)
            steps[2] += 1

        # Screen Drawing
        screen.fill((15, 15, 20))
        env_ql.render_to_surface(surfaces[0])
        env_dqn.render_to_surface(surfaces[1])
        env_neat.render_to_surface(surfaces[2])
        
        agents_data = [
            ("Q-Learning", surfaces[0], 10, m_ql, scores[0], steps[0], dones[0]),
            ("DQN", surfaces[1], VIEW_WIDTH + 20, m_dqn, scores[1], steps[1], dones[1]),
            ("NEAT", surfaces[2], VIEW_WIDTH * 2 + 30, m_neat, scores[2], steps[2], dones[2])
            ]
        
        for name, surf, x_pos, metrics, score, step, done in agents_data:
                # Draw game frame
                screen.blit(surf, (x_pos, 10))
                # Draw HUD card
                hud_rect = pygame.Rect(x_pos, VIEW_HEIGHT + 20, VIEW_WIDTH, PANEL_HEIGHT - 30)
                pygame.draw.rect(screen, (30, 30, 40), hud_rect)
                pygame.draw.rect(screen, (70, 70, 90), hud_rect, 1)
        
                title = header_font.render(f"[{name}]", True, (255, 255, 255))
                live_txt = font.render(f"Live Score: {score} | Moves: {step}", True, (100, 255, 100))
                status_txt = font.render(f"Status: {'FINISHED' if done else 'PLAYING'}", True, (255, 100, 100) if done else (0, 200, 255))
                train_avg = font.render(f"Train Avg: {metrics.get('final_avg_score', 'N/A')}", True, (200, 200, 200))
                train_max = font.render(f"Train Max: {metrics.get('max_score', 'N/A')}", True, (200, 200, 200))
        
                screen.blit(title, (x_pos + 10, VIEW_HEIGHT + 25))
                screen.blit(live_txt, (x_pos + 10, VIEW_HEIGHT + 45))
                screen.blit(status_txt, (x_pos + 10, VIEW_HEIGHT + 65))
                screen.blit(train_avg, (x_pos + 10, VIEW_HEIGHT + 85))
                screen.blit(train_max, (x_pos + 10, VIEW_HEIGHT + 105))
        
        pygame.display.flip()

        # print(states, steps, scores, dones)
        
         # Terminate when all agents complete their session
        if all(dones):
                pygame.time.delay(20000)
                break
        
    pygame.quit()
    sys.exit()
        
if __name__ == '__main__':
    run_arena("snake")