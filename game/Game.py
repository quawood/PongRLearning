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
    trainingIterations = 0
    loading = False
    training = False
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
            for i in range(len(self.players)):
                self.players[i].save('game/q_best/q_weights%d' % i, 'game/q_best/q_biases%d' % i)
            self.isRunning = False

        elif event.type == pygame.KEYDOWN:
            if not self.players[1].isAi:
                if event.key == pygame.K_UP:
                    self.players[1].dir = 1
                elif event.key == pygame.K_DOWN:
                    self.players[1].dir = -1
            if not self.players[0].isAi:
                if event.key == pygame.K_w:
                    self.players[0].dir = 1
                elif event.key == pygame.K_s:
                    self.players[0].dir = -1
            if event.key == pygame.K_RETURN:
                if not self.isPlaying:
                    self.isPlaying = True
            elif event.key == pygame.K_SPACE:
                if not self.training:
                    self.training = True

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

        startHeight = (self.height + self.players[0].height) / 2
        self.players[0].pos = (1, startHeight)
        self.players[1].pos = (self.width - self.players[1].width - 1, startHeight)
        self.players[0].scored = False
        self.players[1].scored = False

        self.isPlaying = True

    def get_features(self, player):
        return np.array(
            [[player.pos[1] / self.height, self.ball.pos[0] / self.width, self.ball.pos[1] / self.height, self.ball.vel[0] / self.ball.speed,
             self.ball.vel[1] / self.ball.speed]])

    def move_players(self):
        scored = False
        p = 0

        currentFeatures = [self.get_features(self.players[0]), self.get_features(self.players[1])]
        self.move_ball()
        for player in self.players:
            if player.scored:
                scored = True

            if player.isAi:
                if player.isTraining:
                    self.train(p, 0.2, currentFeatures[p])
                else:
                    player.move(features=self.get_features(player))
            else:
                player.move(direction=player.dir)

            p += 1
        self.players[0].hit = False
        self.players[1].hit = False

        if scored:
            self.start()

    def train(self, i, epsilon, features):
        player = self.players[i]
        opponent = self.players[i - 1]

        currentFeatures = features
        rand = random.uniform(0, 1)
        if rand < epsilon:
            a = random.randint(0, 2)
            player.move(direction=-(a - 1))
        else:
            player.move(features=currentFeatures)
        a = -player.dir + 1

        newFeatures = self.get_features(player)

        reward = player.agentAI.livingReward
        if not player.prevDir == player.dir:
            reward += 0
        if player.scored:
            reward = 100
            player.agentAI.to_exit(features, reward)
            return
        elif opponent.scored:
            reward = -500
            player.agentAI.to_exit(features, reward)
            return
        if player.hit:
            reward = 50

        sample = (currentFeatures, a, reward, newFeatures)

        player.agentAI.updateQ(sample)
        return

    def move_ball(self):
        ballNextPos = (self.ball.pos[0] + self.ball.vel[0], self.height - (self.ball.pos[1] + self.ball.vel[1]))
        ballRect = pygame.Rect(ballNextPos, (1, 1))

        pNum = 0
        for player in self.players:
            if player.rect.contains(ballRect):
                yIntersect = -(ballNextPos[1] - self.height)
                relYIntersect = yIntersect - player.pos[1] + player.height / 2
                normRelYIntersect = 2 * relYIntersect / player.height
                angle = normRelYIntersect * self.ball.max_angle

                player.hit = True
                if pNum == 0:
                    self.ball.dir = angle
                else:
                    self.ball.dir = math.pi - angle
                self.ball.move()
                return
            pNum += 1

        if ballNextPos[0] > self.width:
            self.players[0].score += 1
            self.players[0].scored = True
        elif ballNextPos[0] < 0:
            self.players[1].score += 1
            self.players[1].scored = True
        elif ballNextPos[1] > self.height or ballNextPos[1] < 0:
            self.ball.dir = -self.ball.dir

        self.ball.move()

    def draw(self):
        self.gameDisplay.fill((0, 0, 0))

        for player in self.players:
            pygame.draw.rect(self.gameDisplay, player.color, player.rect)

        pygame.draw.circle(self.gameDisplay, self.ball.color, (int(self.ball.pos[0]), self.height - int(self.ball.pos[1])),
                           self.ball.radius)

        scoreLabel0 = self.font.render('%d' % (self.players[0].score % 10), 1, (255, 255, 255))
        scoreLabel1 = self.font.render('%d' % (self.players[1].score % 10), 1, (255, 255, 255))
        self.gameDisplay.blit(scoreLabel0, (self.width / 2 - 40 - 30, 5))
        self.gameDisplay.blit(scoreLabel1, (self.width / 2 + 30, 5))

