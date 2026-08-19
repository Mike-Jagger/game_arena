# Game Arena Project

## Project Structure

```text
game_arena/
├── README.md
├── arena_play.py
├── train.py
├── games/
│   ├── base_game.py
│   ├── snake.py
│   ├── tictactoe.py
│   └── connect4.py
├── models/
│   ├── qlearning.py
│   ├── dqn.py
│   ├── neat.py
│   └── ppo.py
├── storage/
│   └── [saved models and training metrics]
└── docs/
    └── [project proposal and supporting documentation]
```

The project is organized so that the game environments, learning models, and scripts are kept separate.

- `README.md` contains the project documentation and explains how the different parts of the system work.
- `arena_play.py` runs the Game Arena and brings the games and trained models together for visual comparison.
- `train.py` handles model training and stores the resulting metrics and model artifacts.
- `games/` contains the game environments. Each game follows the common game interface so that it can be used by the execution scripts.
- `models/` contains the AI agents, such as Q-learning, DQN, NEAT, and PPO implementations.
- `storage/` contains saved model artifacts and training metrics.
- `docs/` contains the project proposal and other supporting documentation.

The main flow is:

**Game Environment → Model/Agent → Training → Saved Model & Metrics → Game Arena (`arena_play.py`) → Visual Comparison**

---

## 1. Architecture Overview

The Game Arena is built on a modular, decoupled architecture. The main idea is to keep the game logic separate from the machine learning algorithms. This makes it possible to use different models with different games without having to rewrite the entire system.

The system consists of three primary layers:

1. **Game Environments (`games/`)**: Standardized classes that handle state management, game rules, rewards, and rendering.
2. **Learning Models (`models/`)**: Standardized agent classes that observe states, select actions, and learn from rewards.
3. **Execution Scripts (`train.py` and `arena_play.py`)**: These connect the environments and models. `train.py` handles training, while `arena_play.py` runs the trained agents and displays the games and performance information.

This project is based on the proposal found in the `/docs` section of the project folder. The main problem identified in the proposal is that comparing the performance of different AI models can be difficult to communicate to non-technical audiences. Results shown only as tables of numbers do not always make it easy to understand why one model performs better than another.

The solution is to build a Game Arena where different AI models can play different games while their training, gameplay, and performance can be observed in a more visual and understandable way.

---

## 2. The Game Environment Interface

To allow the execution scripts to work with different games in the same way, every new game must implement a standard interface inspired by OpenAI Gymnasium.

### Required Game Methods

| Method Signature | Description | Return Value |
| :--- | :--- | :--- |
| `__init__(self, width, height)` | Initializes the game configuration and visual bounds. | `None` |
| `reset(self)` | Resets the game to its initial state. | `state` (Tuple or NumPy Array) |
| `step(self, action)` | Advances the game by one frame or turn based on the agent's action. | `(next_state, reward, done, score_or_info)` |
| `get_state(self)` | Calculates and returns the current state representation. | `state` (Tuple or NumPy Array) |
| `render_to_surface(self, surface)` | Draws the current game frame onto the provided Pygame surface. | `None` |

### Base Template: `games/base_game.py`

The following is the conceptual base template that shows the structure expected from a game environment:

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
        # Apply action, calculate reward, and check terminal state
        # Return next_state, reward, done, info
        pass

    def get_state(self):
        # Return a numerical representation of the environment
        pass

    def render_to_surface(self, surface: pygame.Surface):
        # Draw shapes, sprites, or lines to the given surface
        pass
```

Each individual game is responsible for implementing these methods according to its own rules. The important part is that the execution scripts can call the same methods regardless of which game is being played.

---

## 3. Model Interface

All AI agents, whether they are tabular methods such as Q-learning, deep learning methods such as DQN, or evolutionary methods such as NEAT, should expose a consistent set of methods.

This allows the execution scripts to train the agents and request actions without needing to know the internal details of each model.

### Required Agent Methods

| Method Signature | Description |
| :--- | :--- |
| `__init__(self, ...)` | Initializes the model's hyperparameters, architecture, and memory where required. |
| `get_action(self, state, is_training)` | Returns an action. When `is_training=True`, the agent can use exploration such as epsilon-greedy. |
| `save(self, filepath)` | Saves the model weights or Q-table to the specified location. |
| `load(self, filepath)` | Loads previously saved model weights or Q-table data. |

Training methods such as `update`, `remember`, or `train_step` can be different for each model because different algorithms learn in different ways.

For future extensions, a standard method such as the following could be used to make training more consistent:

```python
train_on_transition(state, action, reward, next_state, done)
```

---

## 4. Extension Guide: Implementing a New Game

To add a new game, such as Connect-4 or another custom game, follow these steps.

### Step 1: Create the File

Create a new Python file inside the `games/` directory.

For example:

```text
games/connect4.py
```

### Step 2: Define the Game Class

Create a class for the new game that implements all the required methods from the Game Interface.

For example:

```python
class Connect4Game:
    ...
```

### Step 3: Design the State Space

The `get_state()` method should return a flat array or tuple that the learning models can easily process.

### Step 4: Implement Rendering

The `render_to_surface()` method should draw the game using the supplied Pygame surface. The rendering should use the provided width and height so that the game fits correctly inside the multi-viewport layout used by `arena_play.py`.

---

## 5. Extension Guide: Implementing a New Model

To add a new learning algorithm, such as Proximal Policy Optimization (PPO), follow these steps.

### Step 1: Create the File

Create a new Python file inside the `models/` directory.

For example:

```text
models/ppo.py
```

### Step 2: Define the Agent

Create an agent class that implements the required methods, including `get_action`, `save`, and `load`.

For example:

```python
class PPOAgent:
    ...
```

### Step 3: Handle Device Placement

If the model uses PyTorch, the model should handle CPU/GPU placement inside its initialization and loading methods so that it can run on the available device.

---

## 6. Integrating Additions into the Architecture

Once a new game or model has been implemented, it must be connected to the command-line interfaces.

### Step 1: Update `train.py`

First, import the new game or model.

Then add its name to the appropriate `argparse` choices.

For example:

```python
parser.add_argument(
    '--game',
    choices=['snake', 'tictactoe', 'connect4'],
    required=True
)

parser.add_argument(
    '--model',
    choices=['qlearning', 'dqn', 'neat', 'ppo'],
    required=True
)
```

A dedicated training loop can then be created for combinations that need model-specific training logic.

For example:

```python
def train_connect4_ppo():
    ...
```

The training process should append the relevant metrics to the `storage/` directory in JSON format.

### Step 2: Update `arena_play.py`

Import the new game or model.

If a new game is being added, create the appropriate conditional block to initialize the new game environment.

If a new model is being added, instantiate the model, load its saved artifact from `storage/`, and add it to the execution and rendering loop.

The relevant metrics, such as `final_avg_score` and maximum score, should also be extracted so that they can be displayed on the arena HUD.

---

## 7. How the Main Files Work Together

The main files work together in the following order:

1. **Game files in `games/`** define the environment, rules, state, rewards, and rendering.
2. **Model files in `models/`** define how the AI observes the game state and chooses actions.
3. **`train.py`** connects a game to a model and runs the training process.
4. **`storage/`** receives the saved model artifacts and training metrics produced during training.
5. **`arena_play.py`** loads the trained models and game environments, runs them, and displays their gameplay and performance.
6. **`README.md`** documents the architecture, interfaces, project structure, and extension process.

This separation makes it easier to add another game or AI model without changing unrelated parts of the project.

---

## 8. Development Workflow

For the two implementations, each feature/documentation change should be developed on its own branch and merged into `main` before moving on to the next implementation.

### Game Arena Implementation

Create the feature branch:

```bash
git checkout -b "feat/game arena based on initial doc"
```

Implement and test the Game Arena changes, then merge the completed branch into `main`.

### Documentation Implementation

Create the documentation branch:

```bash
git checkout -b "doc/initial doc"
```

Complete the README documentation changes, then merge the branch into `main`.

The expected workflow is therefore:

```text
main
 │
 ├── feat/game arena based on initial doc
 │        │
 │        └── merge → main
 │
 └── doc/initial doc
          │
          └── merge → main
```

The repository should finish with the completed work merged into a single `main` branch before moving to the next implementation.

---

## 9. Future Reference

One possible future improvement is to render parts of the Markdown documentation directly inside Pygame. This could allow algorithms and game explanations to be displayed using the README content without requiring the reviewer to hard-code the text separately.

Reference:

https://share.google/aimode/ypUXJbx7uSTyp02rr
