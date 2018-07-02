from core.NeuralNet import NeuralNetwork
import numpy as np

neuralNet = NeuralNetwork([2, 3, 1])
X = np.array([[1, 2], [1, 3], [3, 6], [2, 4]])
y = np.array([[5], [7], [15], [10]])

neuralNet.train(X, y, 1000)
yHat = neuralNet.predict(np.array([[1, 2]]))
print(yHat)