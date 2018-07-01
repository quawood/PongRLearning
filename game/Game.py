import pygame
import random
import math
from game.Player import Player
from game.Ball import Ball


class Game:
    width = 500
    height = 300
    isRunning = True
    isPlaying = False

    def __init__(self):
        self.players = [Player(), Player()]
        self.ball = Ball()

        pygame.init()

        self.gameDisplay = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = self.font = pygame.font.SysFont("monospace", 5)

        self.start()

    def check(self, event):

        if event.type == pygame.QUIT:
            self.isRunning = False

        elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.players[1].dir = 1
                elif event.key == pygame.K_DOWN:
                    self.players[1].dir = -1
                elif event.key == pygame.K_w:
                    self.players[0].dir = 1
                elif event.key == pygame.K_s:
                    self.players[0].dir = -1
                elif event.key == pygame.K_RETURN:
                    if not self.isPlaying:
                        self.isPlaying = True

        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                self.players[1].dir = 0
            elif event.key == pygame.K_w or event.key == pygame.K_s:
                self.players[0].dir = 0

    def start(self):
        print(self.players[0].score)
        print(self.players[1].score)
        self.isPlaying = False
        rand = random.uniform(-1.30, 1.30)
        randSide = random.choice([0, math.pi])
        self.ball.dir = rand + randSide

        self.ball.pos = (self.width / 2, self.height / 2)

        startHeight = (self.height - self.players[0].height) / 2
        self.players[0].pos = (1, startHeight)
        self.players[1].pos = (self.width - self.players[1].width - 1, startHeight)

    def move_player(self):
        for player in self.players:
            nextPos = (player.pos[0], player.pos[1] - player.dir * player.speed)
            if nextPos[1] >= 0 and nextPos[1] <= self.height - player.height:
                player.move(player.dir)

    def move_ball(self):
        for player in self.players:
            if player.rect.contains(self.ball.rect):
                self.ball.dir = -self.ball.dir + math.pi
                self.ball.move()
                return

        if self.ball.pos[0] > self.width:
            self.players[1].score += 1
            self.start()
        elif self.ball.pos[0] < 0:
            self.players[0].score += 1
            self.start()
        elif self.ball.pos[1] > self.height or self.ball.pos[1] < 0:
            self.ball.dir = -self.ball.dir

        self.ball.move()

    def draw(self):
        self.gameDisplay.fill((0, 0, 0))

        for player in self.players:
            pygame.draw.rect(self.gameDisplay, player.color, player.rect)

        pygame.draw.circle(self.gameDisplay, self.ball.color, (int(self.ball.pos[0]), int(self.ball.pos[1])),
                       self.ball.radius)
