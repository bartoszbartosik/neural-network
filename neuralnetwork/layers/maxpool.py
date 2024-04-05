from typing import Callable

import numpy as np

from neuralnetwork.layers import Layer
from neuralnetwork.activations import linear


class MaxPooling(Layer):

    def __init__(self, pool_size: tuple, stride: int):
        super().__init__(activation=linear)
        self.pool_size = pool_size
        self.stride = stride


    def build(self, a_: np.ndarray, loss: Callable) -> None:
        batch_size, rows, cols, channels = a_.shape
        p_rows, p_cols = self.pool_size
        out_rows = (rows - p_rows) // self.stride + 1
        out_cols = (cols - p_cols) // self.stride + 1
        self.shape = (batch_size, out_rows, out_cols, channels)
        self.a = np.zeros(self.shape)


    def feedforward(self, a_: np.ndarray) -> None:
        batch_size, out_rows, out_cols, channels = self.shape
        p_rows, p_cols = self.pool_size

        for b in range(batch_size):
            for c in range(channels):
                for row in range(out_rows):
                    for col in range(out_cols):
                        self.a[b, row, col, c] = np.max(a_[b, row*self.stride : row*self.stride + p_rows, col*self.stride : col*self.stride + p_cols, c])


    def backpropagate(self, grad: np.ndarray, lin: Layer, lout: Layer = None) -> tuple:
        pass

