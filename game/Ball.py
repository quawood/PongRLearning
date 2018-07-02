import math
import pygame


class Ball:
    color = (255, 255, 255)

    def __init__(self):
        self.pos = (0, 0)
        self.dir = 0
        self.speed = 13
        self.radius = 6

    def move(self):
        vel = self.vel
        self.pos = (self.pos[0] - vel[0], self.pos[1] - vel[1])

    @property
    def vel(self):
        return self.speed * math.cos(self.dir), self.speed * math.sin(self.dir)

    @property
    def rect(self):
        return pygame.Rect(self.pos[0] - 0.5, self.pos[1] - 0.5, 1, 1)