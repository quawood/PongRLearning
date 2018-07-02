import pygame
import random
import math
import numpy as np
from game.Player import Player
from game.Ball import Ball
from sklearn.preprocessing import normalize


class Game:
    width = 500
    height = 300
    isRunning = True
    isPlaying = False
    isAutoPilot = True
    trainingIterations = 0

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
            if not self.isAutoPilot:
                if event.key == pygame.K_UP:
                    self.players[1].dir = 1
                elif event.key == pygame.K_DOWN:
                    self.players[1].dir = -1

            if event.key == pygame.K_RETURN:
                if not self.isPlaying:
                    self.isPlaying = True
            elif event.key ==  pygame.K_a:
                self.isAutoPilot = not self.isAutoPilot
            elif event.key == pygame.K_SPACE:
                self.trainingIterations = 0

        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                self.players[1].dir = 0
            elif event.key == pygame.K_w or event.key == pygame.K_s:
                self.players[0].dir = 0

    def start(self):
        self.trainingIterations += 1
        self.isPlaying = False
        rand = random.uniform(math.pi / 5, 1)
        randVert = random.choice([-1, 1])
        randSide = random.choice([0, math.pi])
        self.ball.dir = (rand * randVert) + randSide

        self.ball.pos = (self.width / 2, self.height / 2)

        startHeight = (self.height - self.players[0].height) / 2
        self.players[0].pos = (1, startHeight)
        self.players[1].pos = (self.width - self.players[1].width - 1, startHeight)
        self.players[0].scored = False
        self.players[1].scored = False

        self.isPlaying = True

    def move_player(self):
        if self.isAutoPilot:
            self.train(1, 0.05)
        else:
            for player in self.players:
                nextPos = (player.pos[0], player.pos[1] - player.dir * player.speed)
                if nextPos[1] >= 0 and nextPos[1] <= self.height - player.height:
                    player.move(player.dir)

        self.train(0, 0.05)

        self.players[0].hit = False
        self.players[1].hit = False

    def train(self, n, epsilon):
        player = self.players[n]
        opponent = self.players[1 - n]
        ball = self.ball

        sample = ()  # initialize a sample

        features = normalize(np.array(
            [[player.pos[0], player.pos[1], opponent.pos[0], opponent.pos[1], ball.pos[0], ball.pos[1], ball.vel[0],
             ball.vel[1]]]))

        qActionValues = player.agentAI.qApproximate.predict(features)
        rand = random.uniform(0, 1)
        if rand < epsilon:
            a = random.randint(0,2)
        else:
            a = np.argmax(qActionValues)

        nextPos = (player.pos[0], player.pos[1] + (a - 1) * player.speed)
        if nextPos[1] >= 0 and nextPos[1] <= self.height - player.height:
            a = a
        else:
            a = 1

        player.move(-(a - 1))  # move player with action a
        newFeatures = normalize(np.array([
            [player.pos[0], player.pos[1], opponent.pos[0], opponent.pos[1], ball.pos[0], ball.pos[1], ball.vel[0],
             ball.vel[1]]]))

        reward = player.agentAI.livingReward
        if player.scored:
            reward = 100
            self.start()
            player.agentAI.to_exit(features, reward)
            return
        elif opponent.scored:
            reward = -500
            self.start()
            player.agentAI.to_exit(features, reward)
            return

        if player.hit:
            reward = 50

        sample = (features, a, reward, newFeatures)

        player.agentAI.updateQ(sample)
        return

    def move_ball(self):
        for player in self.players:
            if player.rect.contains(self.ball.rect):
                player.hit = True
                self.ball.dir = -self.ball.dir + math.pi
                self.ball.move()
                return

        if self.ball.pos[0] > self.width:
            self.players[0].score += 1
            self.players[0].scored = True
        elif self.ball.pos[0] < 0:
            self.players[1].score += 1
            self.players[1].scored = True
        elif self.ball.pos[1] > self.height or self.ball.pos[1] < 0:
            self.ball.dir = -self.ball.dir

        self.ball.move()

    def draw(self):
        self.gameDisplay.fill((0, 0, 0))

        for player in self.players:
            pygame.draw.rect(self.gameDisplay, player.color, player.rect)

        pygame.draw.circle(self.gameDisplay, self.ball.color, (int(self.ball.pos[0]), int(self.ball.pos[1])),
                       self.ball.radius)
