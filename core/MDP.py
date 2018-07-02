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

        newPredict = self.qApproximate.predict(newState, static=True)
        predict = self.qApproximate.predict(previousState)
        difference = reward + self.gamma * np.max(newPredict) - predict
        self.qApproximate.updateWeights(-difference, self.alpha)

    def to_exit(self, state, reward):
        predict = self.qApproximate.predict(state)
        difference = reward - predict
        self.qApproximate.updateWeights(-difference, self.alpha)
