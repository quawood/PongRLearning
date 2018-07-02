import numpy as np


class Layer:
    def __init__(self, n):
        self.activation = np.zeros((1, n))
        self.delta = np.zeros((1, n))
        self.z = np.zeros((1, n))
        self.n = n
