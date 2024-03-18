from typing import Callable

import numpy as np
from scipy import signal

from neuralnetwork.layers import Layer


class Convolutional(Layer):

    def __init__(self, kernels: int, kernel_size: tuple, activation: Callable, padding='valid'):
        super().__init__(neurons=10,
                         activation=activation,
                         weights=None,
                         bias=None)

        # Input data
        self.knum = kernels
        self.krows, self.kcols = kernel_size
        self.padding = padding

        # Empty data
        self.biases: np.ndarray = np.array([])
        self.kernels: np.ndarray = np.array([])
        self.shape = ()


    def compile(self, a_: np.ndarray, loss: Callable) -> None:
        self.loss = loss

        # Unpack input axes
        batch_size, rows, cols, channels = a_.shape

        # Compute proper output shape
        match self.padding:
            case 'same': out_rows, out_cols = (rows, cols)
            case 'valid': out_rows, out_cols = (rows - self.krows + 1, cols - self.kcols + 1)
            case _: out_rows, out_cols = (-1, -1)

        # Update layer shape
        self.shape = (batch_size, out_rows, out_cols, self.knum)

        # Initialize kernels
        self.kernels = np.random.rand(self.knum, channels, self.krows, self.kcols) * 2 - 1

        # Initialize biases
        self.biases = np.random.rand(*self.shape[1:])


    def feedforward(self, a_):
        batch_size, _, _, channels = a_.shape
        self.z = np.zeros(self.shape)

        for b in range(batch_size):
            for k in range(self.knum):
                for c in range(channels):
                    self.z[b, :, :, k] += signal.correlate2d(a_[b, :, :, c], self.kernels[k, c, :, :], mode=self.padding) + self.biases[:, :, k]

        self.a = self.activation(self.z)


    def backpropagate(self,
                      l_prev: Layer,
                      delta_prev: np.ndarray = None,
                      l_next: Layer = None,
                      y: np.ndarray = None) -> tuple:
        pass
