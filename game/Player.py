import pygame


class Player:
    width = 10
    height = 100
    color = (255, 255, 255)
    speed = 4

    def __init__(self, pos=None):
        # initialize player
        self.pos = (0, pos)
        self.speed = 8
        self.dir = 0
        self.score = 0

    def move(self, dir):
        self.dir = dir
        self.pos = (self.pos[0], self.pos[1] - dir * self.speed)

    @property
    def rect(self):
        return pygame.Rect(self.pos, (self.width, self.height))
