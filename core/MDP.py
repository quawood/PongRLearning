import numpy as np


class MDP:
    def __init__(self, living=0, gamma=1, alpha=0.1):
        self.qApproximate = None
        self.livingReward = living
        self.gamma = gamma
        self.alpha = alpha

    def updateQ(self, sample):
        previousState = sample[0]
        action = sample[1]
        reward = sample[2]
        newState = sample[3]

        newPredict = self.qApproximate.predict(newState)
        predict = self.qApproximate.predict(previousState)
        difference = reward + self.gamma * np.max(newPredict) - predict[0, action]
        difVector = np.zeros((1, 3))
        difVector[0, action] = difference
        self.qApproximate.update_weights(-difVector, self.alpha)

    def to_exit(self, state, reward):
        predict = self.qApproximate.predict(state)
        difference = reward - predict
        self.qApproximate.update_weights(-difference, self.alpha)
