from typing import Callable

import numpy as np

from neuralnetwork.layers import Layer


class Convolutional(Layer):
    def __init__(self, kernels: int, kernel_size: tuple, activation: Callable, stride, padding):
        super().__init__(neurons=neurons,
                         activation=activation,
                         weights=weights,
                         bias=bias)

        self.kernels: np.ndarray = np.ones(kernels, kernel_size[0], kernel_size[1])

        self.bias = np.zeros([])

    def init_params(self, a_):
        self.kernels = np.expand_dims(self.kernels, axis=0)

    def feedforward(self, a_):
        pass



