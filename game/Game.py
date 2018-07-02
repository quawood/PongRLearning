import pygame
import random
import math
import numpy as np
from game.Player import Player
from game.Ball import Ball
from sklearn.preprocessing import normalize


class Game:
    width = int((3 / 2) * 500)
    height = int((3 / 2) * 300)
    isRunning = True
    isPlaying = False
    trainingIterations = 0
    loading = False

    def __init__(self, loading=False):
        self.players = [Player(self.height), Player(self.height)]
        self.ball = Ball()

        if loading:
            for i in range(len(self.players)):
                self.players[i].load('game/q_best/q_weights%d' % i, 'game/q_best/q_biases%d' % i)

        pygame.init()

        self.gameDisplay = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font("assets/ATARCC__.TTF", 40)

        self.start()

    def check(self, event):
        if event.type == pygame.QUIT:
            self.isRunning = False
        elif event.type == pygame.KEYDOWN:
            if not self.players[0].isAi:
                if event.key == pygame.K_w:
                    self.players[0].dir = 1
                elif event.key == pygame.K_s:
                    self.players[0].dir = -1
            if not self.players[1].isAi:
                if event.key == pygame.K_UP:
                    self.players[1].dir = 1
                elif event.key == pygame.K_DOWN:
                    self.players[1].dir = -1
            if event.key == pygame.K_RETURN:
                if not self.isPlaying:
                    self.isPlaying = True
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

        # choose random direction in range for ball to start with
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
        for player in self.players:
            if player.scored:
                self.start()
                return

            if player.isAi:
                if player.isTraining:
                    self.train(0.05, player)
                else:
                    player.move(self.get_features(player))
            else:
                player.move()

        self.players[0].hit = False
        self.players[1].hit = False

    def get_features(self, player):
        return normalize(np.array(
            [[player.pos[1], self.ball.pos[0], self.ball.pos[1], self.ball.vel[0],
             self.ball.vel[1], self.ball.speed]]))

    def train(self,  epsilon, p):
        player = p
        opponent = [pad for pad in self.players if not pad == p][0]
        ball = self.ball

        features = self.get_features(player)

        # take random action with probability epsilon
        rand = random.uniform(0, 1)
        if rand < epsilon:
            a = random.randint(0, 2)
            player.dir = -(a - 1)
            player.move()
        else:
            player.move(features=features)

        a = -player.dir + 1

        # get new state
        newFeatures = self.get_features(player)

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

        # render scores
        score1Label = self.font.render("%d" % self.players[0].score, 1, (255, 255, 255))
        self.gameDisplay.blit(score1Label, (self.width/2 - 45, 10))
        score2Label = self.font.render("%d" % self.players[1].score, 1, (255, 255, 255))
        self.gameDisplay.blit(score2Label, (self.width / 2 + 5, 10))