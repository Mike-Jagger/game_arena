"""
Tic-Tac-Toe environment.

Board is a 9-cell tuple:
  0 = empty, 1 = agent's mark (X), -1 = opponent's mark (O)
Index layout:
  0 1 2
  3 4 5
  6 7 8

Using a tuple (not a list) as the state is important: Q-learning needs
states to be hashable so they can be dictionary keys in the Q-table.
"""

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
    (0, 4, 8), (2, 4, 6),              # diagonals
]


class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [0] * 9
        self.done = False
        return self.get_state()

    def get_state(self):
        return tuple(self.board)

    def available_actions(self):
        return [i for i, cell in enumerate(self.board) if cell == 0]

    def check_winner(self):
        for a, b, c in WIN_LINES:
            total = self.board[a] + self.board[b] + self.board[c]
            if total == 3:
                return 1       # agent (X) wins
            if total == -3:
                return -1      # opponent (O) wins
        if 0 not in self.board:
            return 0           # draw
        return None             # game not over yet

    def step(self, action, mark):
        """
        Place `mark` (1 or -1) at `action` (0-8).
        Returns (state, reward, done) — reward is always from the AGENT's
        (mark=1) point of view, since that's the perspective we train.
        """
        if self.board[action] != 0:
            raise ValueError(f"Cell {action} is already taken")

        self.board[action] = mark
        result = self.check_winner()

        if result is None:
            reward, done = 0.0, False
        elif result == 1:
            reward, done = 1.0, True     # agent won
        elif result == -1:
            reward, done = -1.0, True    # agent lost
        else:
            reward, done = 0.5, True     # draw — better than losing, worse than winning

        self.done = done
        return self.get_state(), reward, done

    def render(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        rows = [self.board[i:i + 3] for i in range(0, 9, 3)]
        for row in rows:
            print(" ".join(symbols[c] for c in row))
        print()
