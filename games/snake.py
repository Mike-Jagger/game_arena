# This is hugely based on this implementation of an AI Snake game: https://github.com/SandeepPadhi/AI_That_Plays_Snake

import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

Point = namedtuple('Point', 'x, y')
BLOCK_SIZE = 20

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class SnakeGame:
    def __init__(self, width=400, height=400):
        self.w = width
        self.h = height
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        return self.get_state()

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def step(self, action):
        """Actions: [staright, right, left] (one-hot or 0, 1, 2)"""
        self.frame_iteration += 1

        # Determine the new direction
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action == 0:
            new_dir = clock_wise[idx]  # no change, continue straight
        elif action == 1:
            new_dir = clock_wise[(idx + 1) % 4]  # right turn
        else:
            new_dir = clock_wise[(idx - 1) % 4]  # left turn

        self.direction = new_dir
        self._move(self.direction)
        self.snake.insert(0, self.head)

        reward = 0 
        game_over = False

        # Check if collision with wall or self, or timeout
        # TODO: number of non successful attempts should be configurable 
        # (looking at the 100) using CLI inputs
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return self.get_state(), reward, game_over, self.score

        # Check if food is eaten
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        return self.get_state(), reward, game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Check if the snake hits the boundaries
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Check if the snake hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def get_state(self):
        head = self.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self._is_collision(point_r)) or 
            (dir_l and self._is_collision(point_l)) or 
            (dir_u and self._is_collision(point_u)) or 
            (dir_d and self._is_collision(point_d)),

            # Danger right
            (dir_u and self._is_collision(point_r)) or 
            (dir_d and self._is_collision(point_l)) or 
            (dir_l and self._is_collision(point_u)) or 
            (dir_r and self._is_collision(point_d)),

            # Danger left
            (dir_d and self._is_collision(point_r)) or 
            (dir_u and self._is_collision(point_l)) or 
            (dir_r and self._is_collision(point_u)) or 
            (dir_l and self._is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.food.x < head.x,  # food left
            self.food.x > head.x,  # food right
            self.food.y < head.y,  # food up
            self.food.y > head.y  # food down
        ]

        return np.array(state, dtype=int)

    def render_to_surface(self, surface):
        """Renders game frame directly onto a provided pygame.Surface."""

        surface.fill((20, 20, 25))
        for pt in self.snake:
            pygame.draw.rect(surface, (0, 200, 80), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(surface, (0, 150, 60), pygame.Rect(pt.x+2, pt.y+2, BLOCK_SIZE-4, BLOCK_SIZE-4))

        # Draw Food
        pygame.draw.rect(surface, (230, 50, 50), pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))