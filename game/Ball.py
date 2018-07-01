import math
import pygame

class Ball:
    color = (255, 255, 255)

    def __init__(self):
        self.pos = (0, 0)
        self.dir = 0
        self.speed = 10
        self.radius = 5

    def move(self):
        self.pos = (self.pos[0] - self.speed * math.cos(self.dir), self.pos[1] - self.speed * math.sin(self.dir))

    @property
    def rect(self):
        return pygame.Rect(self.pos[0] - 0.5, self.pos[1] - 0.5, 1, 1)