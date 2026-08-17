import pygame
import numpy as np
import random

class TicTacToeGame:
    def __init__(self, size=300):
        self.size = size
        self.cell_size = size // 3
        self.reset()

    def reset(self):
        self.board = np.zeros(9, dtype=int)  # 0: empty, 1: AI (X), -1: Opponent (O)
        self.current_player = 1
        self.winner = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        return tuple(self.board)

    def get_available_actions(self):
        return [i for i, cell in enumerate(self.board) if cell == 0]

    def step(self, action):
        if self.board[action] != 0 or self.done:
            return self.get_state(), -10, True, {"result": "illegal"}

        self.board[action] = 1
        if self.check_win(1):
            self.done = True
            return self.get_state(), 10, True, {"result": "win"}
        if len(self.get_available_actions()) == 0:
            self.done = True
            return self.get_state(), 1, True, {"result": "draw"}

        # Opponent move (random baseline or rule-based)
        opp_action = random.choice(self.get_available_actions())
        self.board[opp_action] = -1
        if self.check_win(-1):
            self.done = True
            return self.get_state(), -10, True, {"result": "loss"}
        if len(self.get_available_actions()) == 0:
            self.done = True
            return self.get_state(), 1, True, {"result": "draw"}

        return self.get_state(), 0, False, {"result": "ongoing"}

    def check_win(self, player):
        b = self.board.reshape(3, 3)
        for i in range(3):
            if np.all(b[i, :] == player) or np.all(b[:, i] == player):
                return True
        if b[0, 0] == b[1, 1] == b[2, 2] == player or b[0, 2] == b[1, 1] == b[2, 0] == player:
            return True
        return False

    def render_to_surface(self, surface):
        surface.fill((30, 30, 35))
        # Draw grid
        for i in range(1, 3):
            pygame.draw.line(surface, (100, 100, 100), (0, i * self.cell_size), (self.size, i * self.cell_size), 3)
            pygame.draw.line(surface, (100, 100, 100), (i * self.cell_size, 0), (i * self.cell_size, self.size), 3)

        # Draw tokens
        for i, val in enumerate(self.board):
            r, c = i // 3, i % 3
            center = (c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2)
            if val == 1:  # X
                offset = self.cell_size // 3
                pygame.draw.line(surface, (50, 150, 255), (center[0] - offset, center[1] - offset), (center[0] + offset, center[1] + offset), 4)
                pygame.draw.line(surface, (50, 150, 255), (center[0] + offset, center[1] - offset), (center[0] - offset, center[1] + offset), 4)
            elif val == -1:  # O
                pygame.draw.circle(surface, (255, 80, 80), center, self.cell_size // 3, 4)