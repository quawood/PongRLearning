import numpy as np
from core.Layer import Layer


def f(x, derive=False):
    if derive:
        return 1
    else:
        return x


def sum_squared_errors(prediction, y):
    difference = prediction - y
    return difference.dot(difference.T)[0][0]


class NeuralNetwork:
    def __init__(self, topology):

        layers = []
        for nNode in topology:
            layers.append(Layer(nNode))
        self.layers = layers

        self.weightsSet = []
        self.biasSet = []
        for n in range(0, len(topology) - 1):
            self.weightsSet.append(2 * np.random.rand(topology[n], topology[n + 1]) - 1)
            self.biasSet.append(2 * np.random.rand(1, topology[n + 1]) - 1)
        self.f = f

        self.previousError = 0

    def predict(self, X):
        # X is m x nFeatures shape
        inputLayer = self.layers[0]
        inputLayer.activation = X
        inputLayer.z = X

        for weightsInd in range(len(self.weightsSet)):
            z = self.layers[weightsInd].activation.dot(self.weightsSet[weightsInd]) + self.biasSet[weightsInd]
            activation = f(z)
            self.layers[weightsInd + 1].activation = activation

        return self.layers[-1].activation

    def update_weights(self, difference, alpha):
        self.layers[-1].delta = difference

        for ind in range(len(self.weightsSet), 0, -1):
            weightsInd = ind - 1
            currentLayer = self.layers[weightsInd + 1]
            nextLayer = self.layers[weightsInd]

            self.layers[weightsInd].delta = currentLayer.delta.dot(self.weightsSet[weightsInd].T)
            self.weightsSet[weightsInd] -= alpha * (nextLayer.activation.T.dot(currentLayer.delta))
            self.biasSet[weightsInd] -= alpha * np.sum(currentLayer.delta, axis=0)

    def train(self, X, y, max_iterations):
        iteration = 0
        while iteration < max_iterations:
            predict = self.predict(X)
            difference = predict - y

            self.update_weights(difference, 0.001)
            iteration += 1
