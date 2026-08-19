import pygame
import random
import numpy as np

class FlappyBirdGame:
    def __init__(self, width=300, height=300):
        self.width = width
        self.height = height
        self.bird_x = int(self.width * 0.2)
        self.gravity = 1.0
        self.jump_strength = -10.0
        self.pipe_width = 40
        self.pipe_gap = 100
        self.pipe_speed = 5
        self.reset()

    def reset(self):
        self.bird_y = self.height // 2
        self.velocity = 0.0
        self.pipes = []
        self.score = 0
        self.frames = 0
        self.done = False
        self._spawn_pipe()
        return self.get_state()

    def _spawn_pipe(self):
        min_y = 50
        max_y = self.height - 50 - self.pipe_gap
        top_height = random.randint(min_y, max_y)
        self.pipes.append({
            'x': self.width,
            'top': top_height,
            'bottom': top_height + self.pipe_gap,
            'passed': False
        })

    def step(self, action):
        """Action 0: Do nothing, Action 1: Flap"""
        if self.done:
            return self.get_state(), 0, True, self.score

        self.frames += 1
        reward = 0.1  # Reward for staying alive

        if action == 1:
            self.velocity = self.jump_strength

        self.velocity += self.gravity
        self.bird_y += int(self.velocity)

        # Move pipes
        for pipe in self.pipes:
            pipe['x'] -= self.pipe_speed

        # Remove off-screen pipes
        if self.pipes and self.pipes[0]['x'] < -self.pipe_width:
            self.pipes.pop(0)

        # Spawn new pipes
        if self.pipes and self.pipes[-1]['x'] < self.width - 150:
            self._spawn_pipe()

        # Check collisions
        hit_ground_or_ceiling = self.bird_y >= self.height or self.bird_y <= 0
        hit_pipe = False
        next_pipe = self.pipes[0] if self.pipes else None

        if next_pipe:
            # Check horizontal bounds
            if next_pipe['x'] <= self.bird_x <= next_pipe['x'] + self.pipe_width:
                # Check vertical bounds
                if self.bird_y <= next_pipe['top'] or self.bird_y >= next_pipe['bottom']:
                    hit_pipe = True

            # Score update
            if not next_pipe['passed'] and next_pipe['x'] + self.pipe_width < self.bird_x:
                next_pipe['passed'] = True
                self.score += 1
                reward = 10.0

        if hit_ground_or_ceiling or hit_pipe:
            self.done = True
            reward = -10.0

        return self.get_state(), reward, self.done, self.score

    def get_state(self, discrete=False):
        """
        Returns a 4D state: vertical distance to top pipe, vertical distance to bottom pipe,
        horizontal distance to next pipe, and current velocity.
        """
        next_pipe = None
        for pipe in self.pipes:
            if pipe['x'] + self.pipe_width > self.bird_x:
                next_pipe = pipe
                break

        if not next_pipe:
            return np.array([0, 0, 0, self.velocity], dtype=np.float32)

        dist_x = next_pipe['x'] - self.bird_x
        dist_y_top = self.bird_y - next_pipe['top']
        dist_y_bottom = next_pipe['bottom'] - self.bird_y

        state = [dist_x, dist_y_top, dist_y_bottom, self.velocity]

        if discrete:
            # Discretize into roughly 10x10x10x10 grid for Tabular Q-Learning
            # Tabular Q-Learning requires a finite number of recurring states to build its Q-table.
            # Since our distances and velocities are continuous floats, every frame would be a "new" state.
            # We discretize by dividing by 20 (creating "bins") so similar positions are treated as the exact same state.
            return tuple([int(s // 20) for s in state])
            
        return np.array(state, dtype=np.float32)

    def render_to_surface(self, surface):
        surface.fill((135, 206, 235))  # Sky blue
        
        # Draw Bird
        pygame.draw.circle(surface, (255, 200, 0), (self.bird_x, int(self.bird_y)), 12)
        
        # Draw Pipes
        for pipe in self.pipes:
            # Top pipe
            pygame.draw.rect(surface, (0, 200, 50), (pipe['x'], 0, self.pipe_width, pipe['top']))
            # Bottom pipe
            pygame.draw.rect(surface, (0, 200, 50), (pipe['x'], pipe['bottom'], self.pipe_width, self.height - pipe['bottom']))