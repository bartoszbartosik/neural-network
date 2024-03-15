from abc import ABC
from typing import Callable

import numpy as np


class Layer(ABC):

    def __init__(self, neurons: int, activation: Callable, weights: np.ndarray = None, bias: float = None):
        self.input_data: np.ndarray = np.zeros([])
        self.neurons = neurons

        self.weights: np.ndarray = weights
        self.bias: float = bias

        self.activation = activation
        self.values: np.ndarray = np.zeros(neurons)


    def init_params(self):
        if self.weights is None:
            self.weights = np.random.rand(self.neurons, len(self.input_data))
        if self.bias is None:
            self.bias = np.random.random()*2 - 1


    def feedforward(self):
        self.values = self.activation(np.dot(self.weights, self.input_data) + self.bias)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # #   O V E R W R I T T E N   F U N C T I O N S   # # # # # # # # # # # # # # # # #
    def __str__(self) -> str:
        return 'Layer: [values: {}]'.format(self.values)

    def __repr__(self):
        return self.__str__()

