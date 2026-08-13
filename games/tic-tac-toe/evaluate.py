"""
Compares the trained Q-learning agent against a pure Random agent, and
produces the two charts you'll want for the report/demo:

  1. learning_curve.png   - win rate over training episodes (shows the agent improving)
  2. comparison_bar.png   - Random agent vs Q-learning agent, head-to-head final results
"""
import json
import random
import matplotlib.pyplot as plt
from env import TicTacToe
from q_agent import QLearningAgent

N_GAMES = 1000


def random_vs_random(n_games=N_GAMES):
    """Baseline: what does a Random agent achieve against another Random agent?"""
    env = TicTacToe()
    wins = draws = losses = 0
    for _ in range(n_games):
        env.reset()
        done = False
        mark = 1
        while not done:
            action = random.choice(env.available_actions())
            _, reward, done = env.step(action, mark)
            mark *= -1
        if reward == 1.0:
            wins += 1
        elif reward == 0.5:
            draws += 1
        else:
            losses += 1
    return wins / n_games, draws / n_games, losses / n_games


def qlearning_vs_random(agent, n_games=N_GAMES):
    """Trained agent (X, greedy) against a Random opponent (O)."""
    env = TicTacToe()
    wins = draws = losses = 0
    for _ in range(n_games):
        state = env.reset()
        done = False
        while not done:
            actions = env.available_actions()
            action = agent.choose_action(state, actions, greedy=True)
            state, reward, done = env.step(action, mark=1)
            if not done:
                opp_action = random.choice(env.available_actions())
                state, reward, done = env.step(opp_action, mark=-1)
        if reward == 1.0:
            wins += 1
        elif reward == 0.5:
            draws += 1
        else:
            losses += 1
    return wins / n_games, draws / n_games, losses / n_games


def plot_learning_curve(history):
    episodes = [h["episode"] for h in history]
    win = [h["win_rate"] * 100 for h in history]
    draw = [h["draw_rate"] * 100 for h in history]
    loss = [h["loss_rate"] * 100 for h in history]

    plt.figure(figsize=(9, 5.5))
    plt.plot(episodes, win, label="Win rate", color="#0F6E56", linewidth=2)
    plt.plot(episodes, draw, label="Draw rate", color="#854F0B", linewidth=2)
    plt.plot(episodes, loss, label="Loss rate", color="#A32D2D", linewidth=2)
    plt.xlabel("Training episodes")
    plt.ylabel("Rate (%) over 200 evaluation games")
    plt.title("Q-learning agent performance during training\n(playing against a random opponent)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=150)
    plt.close()


def plot_comparison(rand_stats, q_stats):
    labels = ["Win", "Draw", "Loss"]
    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5.5))
    plt.bar([i - width/2 for i in x], [v * 100 for v in rand_stats], width,
            label="Random agent (baseline)", color="#B4B2A9")
    plt.bar([i + width/2 for i in x], [v * 100 for v in q_stats], width,
            label="Q-learning agent (trained)", color="#0F6E56")
    plt.xticks(list(x), labels)
    plt.ylabel(f"Percentage of {N_GAMES} games")
    plt.title("Model comparison: Random vs. Q-learning agent\n(both playing X against a random O opponent)")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("comparison_bar.png", dpi=150)
    plt.close()


def main():
    with open("training_history.json") as f:
        history = json.load(f)

    agent = QLearningAgent(epsilon=0.0)
    agent.load("q_table.pkl")

    print(f"Running {N_GAMES} games for each model...")
    rand_stats = random_vs_random(N_GAMES)
    q_stats = qlearning_vs_random(agent, N_GAMES)

    print("\n=== Final comparison ===")
    print(f"Random agent   -> win {rand_stats[0]:.1%}  draw {rand_stats[1]:.1%}  loss {rand_stats[2]:.1%}")
    print(f"Q-learning     -> win {q_stats[0]:.1%}  draw {q_stats[1]:.1%}  loss {q_stats[2]:.1%}")

    plot_learning_curve(history)
    plot_comparison(rand_stats, q_stats)
    print("\nSaved learning_curve.png and comparison_bar.png")


if __name__ == "__main__":
    main()
