from abc import ABC
from typing import Callable

import numpy as np


class Layer(ABC):

    def __init__(self, neurons: int, activation: Callable, weights: np.ndarray = None, bias: float = None):
        self.n = neurons

        self.w: np.ndarray = weights
        self.b: float = bias

        self.activation = activation
        self.z: np.ndarray = np.zeros(neurons)
        self.a: np.ndarray = np.zeros(neurons)


    def init_params(self, a_):
        if self.w is None:
            self.w = np.random.rand(self.n, len(a_)) * 2 - 1
        if self.b is None:
            self.b = np.random.random() * 2 - 1


    def feedforward(self, a_):
        self.z = (np.dot(self.w, a_.transpose()) + self.b).transpose()
        self.a = self.activation(self.z)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # #   O V E R W R I T T E N   F U N C T I O N S   # # # # # # # # # # # # # # # # #
    def __str__(self) -> str:
        return 'Layer: [{}]'.format(self.a)

    def __repr__(self):
        return self.__str__()

