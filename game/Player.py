import pygame
import numpy as np
from core.MDP import MDP
from core.NeuralNet import NeuralNetwork


class Player:
    width = 20
    height = 100
    color = (255, 255, 255)
    speed = 4

    def __init__(self, gHeight, pos=0):
        # initialize player
        self.gHeight = gHeight
        self.pos = (0, pos)
        self.speed = 12
        self.dir = 0
        self.score = 0
        self.scored = False
        self.hit = False

        self.isTraining = False
        self.isAi = False
        self.agentAI = MDP(0, 1, 0.01)
        self.agentAI.qApproximate = NeuralNetwork([6, 3])

    def move(self, features=None):
        if features is not None:
            qActionValues = self.agentAI.qApproximate.predict(features)
            a = np.argmax(qActionValues)
            direction = - (a - 1)
            self.dir = direction

        nextPos = (self.pos[0], self.pos[1] - self.dir * self.speed)
        if 0 <= nextPos[1] <= self.gHeight - self.height:
            self.pos = nextPos
        else:
            self.dir = 0

    @property
    def rect(self):
        return pygame.Rect(self.pos, (self.width, self.height))

    def save(self, file1, file2):
        with open(file1, 'wb') as f:
            for weight in self.agentAI.qApproximate.weightsSet:
                np.save(f, weight)
        with open(file2, 'wb') as f:
            for bias in self.agentAI.qApproximate.biasSet:
                np.save(f, bias)

    def load(self, file1, file2):
        with open(file1, 'rb') as f:
            for i in range(len(self.agentAI.qApproximate.weightsSet)):
                self.agentAI.qApproximate.weightsSet[i] = np.load(f)
        with open(file2, 'rb') as f:
            for i in range(len(self.agentAI.qApproximate.biasSet)):
                self.agentAI.qApproximate.biasSet[i] = np.load(f)
