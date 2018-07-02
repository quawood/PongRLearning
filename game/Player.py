import pygame
import numpy as np
from core.MDP import MDP
from core.NeuralNet import NeuralNetwork


class Player:
    width = 10
    height = 100
    color = (255, 255, 255)
    speed = 4

    def __init__(self, pos=0):
        # initialize player
        self.pos = (0, pos)
        self.speed = 8
        self.dir = 0
        self.score = 0
        self.scored = False
        self.hit = False

        self.agentAI = MDP(0, 1, 0.01)
        self.agentAI.qApproximate = NeuralNetwork([6, 3])

    def move(self, direc):
        self.dir = direc
        self.pos = (self.pos[0], self.pos[1] - direc * self.speed)

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
