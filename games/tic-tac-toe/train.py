"""
Train a Q-learning agent to play Tic-Tac-Toe (as X) against a random opponent (O).

Every EVAL_EVERY episodes, we pause training and measure the agent's current
win rate against a fresh random opponent (playing greedily, no exploration).
That gives us the learning curve: win rate vs. training progress.
"""
import random
import json
from env import TicTacToe
from q_agent import QLearningAgent

EPISODES = 50_000
EVAL_EVERY = 1_000
EVAL_GAMES = 200


def play_episode(env, agent, training=True):
    """Play one game: agent is X (mark=1), random opponent is O (mark=-1)."""
    state = env.reset()

    while True:
        # --- Agent's turn ---
        actions = env.available_actions()
        action = agent.choose_action(state, actions, greedy=not training)
        next_state, reward, done = env.step(action, mark=1)

        if not done:
            # --- Random opponent's turn ---
            opp_actions = env.available_actions()
            opp_action = random.choice(opp_actions)
            next_state, opp_reward, done = env.step(opp_action, mark=-1)
            reward = opp_reward  # reward is always from the agent's perspective (see env.py)

        if training:
            next_actions = env.available_actions() if not done else []
            agent.update(state, action, reward, next_state, next_actions, done)

        state = next_state
        if done:
            return reward  # 1.0 win, 0.5 draw, -1.0 loss


def evaluate(agent, n_games=EVAL_GAMES):
    env = TicTacToe()
    wins = draws = losses = 0
    for _ in range(n_games):
        result = play_episode(env, agent, training=False)
        if result == 1.0:
            wins += 1
        elif result == 0.5:
            draws += 1
        else:
            losses += 1
    return wins / n_games, draws / n_games, losses / n_games


def main():
    env = TicTacToe()
    agent = QLearningAgent()

    history = []  # list of (episode, win_rate, draw_rate, loss_rate, epsilon)

    for episode in range(1, EPISODES + 1):
        play_episode(env, agent, training=True)
        agent.decay_epsilon()

        if episode % EVAL_EVERY == 0:
            win_rate, draw_rate, loss_rate = evaluate(agent)
            history.append({
                "episode": episode,
                "win_rate": win_rate,
                "draw_rate": draw_rate,
                "loss_rate": loss_rate,
                "epsilon": agent.epsilon,
            })
            print(f"Episode {episode:>6} | epsilon={agent.epsilon:.3f} | "
                  f"win={win_rate:.2%} draw={draw_rate:.2%} loss={loss_rate:.2%}")

    agent.save("q_table.pkl")
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete. Saved q_table.pkl and training_history.json")
    print(f"Q-table has {len(agent.q_table)} learned states "
          f"(out of ~5,478 reachable tic-tac-toe states).")


if __name__ == "__main__":
    main()
