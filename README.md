game_arena/
├── README.md   ← THIS ONE
├── arena_play.py
├── games/
├── models/
└── storage/
# Game Arena Project

## 1. Architecture Overview

The Game Arena is built on a modular, decoupled architecture to ensure that game logic is entirely separate from the machine learning algorithms. This allows for seamless cross-testing (e.g., running the same DQN agent on both Snake and Tic-Tac-Toe).

The system consists of three primary layers:

1. **Game Environments (`games/`)**: Standardized classes that handle state management, rules, rewards, and rendering.
2. **Learning Models (`models/`)**: Standardized agent classes that observe states, select actions, and learn from rewards.
3. **Execution Scripts (`train.py` & `arena_play.py`)**: The orchestrators that tie environments and models together, handling the training loops, metric storage, and simultaneous Pygame visual rendering.

This project is based on the proposal submitted found in the `/docs` section of the project folder. To restate the problem identified:

> Comparing the performance of different AI models (e.g., different reinforcement
> learning algorithms, or the same algorithm with different
> hyperparameters/architectures) is often abstract and hard to communicate to non-
> technical audiences. Benchmarks reported as tables of numbers make it difficult to
> build intuition for why one model outperforms another.

## 2. The Game Environment Interface

To ensure the execution scripts can interact with any game uniformly, all new games must implement a standard interface inspired by OpenAI Gymnasium.

### Required Game Methods

| Method Signature | Description | Return Value |
| :--- | :--- | :--- |
| `__init__(self, width, height)` | Initializes game configuration and visual bounds. | `None` |
| `reset(self)` | Resets the game to its initial state. | `state` (Tuple or Numpy Array) |
| `step(self, action)` | Advances the game by one frame/turn based on the agent's action. | `(next_state, reward, done, score_or_info)` |
| `get_state(self)` | Calculates and returns the current state representation. | `state` (Tuple or Numpy Array) |
| `render_to_surface(self, surface)` | Draws the current game frame directly onto the provided Pygame surface. | `None` |

### Base Template: `games/base_game.py`

*(Conceptual template for reference)*

```python
import pygame

class BaseGame:
    def __init__(self, width=300, height=300):
        self.width = width
        self.height = height
        self.reset()
        
    def reset(self):
        # Reset internal variables
        return self.get_state()
        
    def step(self, action):
        # Apply action, calculate reward, check terminal state
        # return next_state, reward, done, info
        pass
        
    def get_state(self):
        # Return a numerical representation of the environment
        pass
        
    def render_to_surface(self, surface: pygame.Surface):
        # Draw shapes, sprites, or lines to the given surface
        pass

The creative solution we came up with to solve this problem was to build a game arena that would have different models play different games and present their differencies in a much more educational and tangible manner via seeing metrics on training, gameplay and performance differences between the different games as they play them.

The following sections will touch much more on the technical spec, however, starting with the project's architecture

## Project Architecture

## Future ref

Rendering Markdown in pygame to describe algorithms and games using readme and letting the game render readme without having reviewer hardcode stuff: https://share.google/aimode/ypUXJbx7uSTyp02rr


The Model Interface

All Al agents, whether tabular (Q-learning), deep (DQN), or evolutionary (NEAT), must expose a consistent set of methods to allow the execution scripts to train them and query them for actions.

Required Agent Methods

Method Signature                                                            Description

__init__(self, ...)                                                       Initializes hyperparameters, model architecture, and memory.

get actiong self, state, is_training)                                     Returns an action. If is_training=True, incorporates exploration (e.g., epsilon-greedy).

save(self, filepath)                                                      Serializes and saves model weights or Q-tables to disk.

load(self, filepath)                                                      Deserializes and loads model weights or Q-tables from disk.





Note: Training methods (like update or remember + train_step) are currently specific to the model type in train.py, but standardization via a train_on_transition(state, action, reward, next_state, done) method is recommended for future extensions.

4. Extension Guide: Implementing a New Game

To add a new game (e.g., Connect-4 or a custom Mario-style level) [cite: 1], follow these steps:

1. Create the File: Create a new Python file in the games/directory (e.g., games/connect4.py).

2. Define the Class: Create a class (e.g., Connect4Game) that implements all required methods from the Game Interface.

3. Design the State Space: Ensure get_state() returns a flat array or tuple that models can easily ingest.

4. Implement Rendering: The render_to_surface method must scale its drawing based on the width and height parameters so it fits neatly into the multi-viewport arena_play.py screen.


Extension Guide: Implementing a New Model

To add a new learning algorithm (e.g., Proximal Policy Optimization - PPO):

1. Create the File: Create a new file in the models/ directory (e.g., models/ppo.py).

2. Define the Agent: Create a class (e.g., PPOAgent) implementing get_action, save, and load.

3. Handle Device Placement: If using PyTorch, ensure the model dynamically handles CPU/GPU placement inside the __init__ and load methods.



Integrating Additions into the Architecture

Once your new Game or Model is written, you must expose it to the command-line interfaces.

Step 1: Update train.py

1. Import your new game or model.

2. Add the name to the argparse choices.


parser.add_argument('--game', choices=['snake', 'tictactoe', 'connect4'], requis

parser.add_argument('--model', choices=['qlearning', 'don', 'neat', 'ppo'], requ

3. Create a dedicated training loop function (eg, train_connect4_ppo()) that handles the specific memory and optimization steps of your agent, appending metrics to the storage/ directory in JSON format.

Step 2: Update arena_play.py

1 Import your new game or model.

2. If adding a game, add a conditional block to initialize instances of your new game environment.

3. If adding a model, instantiate it, load its saved artifact from storage/, and add it to the execution loop and rendering queue. Ensure you extract the relevant final_avg_score and max score metrics to be displayed on the HUD.









