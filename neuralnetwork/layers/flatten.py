from typing import Callable

import numpy as np

from neuralnetwork.layers import Layer
from neuralnetwork.activations import linear


class Flatten(Layer):

    def __init__(self):
        super().__init__(activation=linear)


    def compile(self, a_: np.ndarray, loss: Callable) -> None:
        batch_size, rows, cols, channels = a_.shape
        self.shape = (batch_size, rows*cols*channels)


    def feedforward(self, a_: np.ndarray) -> None:
        batch_size, out_rows, out_cols, channels = a_.shape
        self.a = np.zeros(self.shape)

        for b in range(batch_size):
            n = 0
            for c in range(channels):
                for row in range(out_rows):
                    for col in range(out_cols):
                        self.a[b, n] = a_[b, row, col, c]
                        n += 1


    def backpropagate(self, grad: np.ndarray, lin: Layer, lout: Layer = None) -> tuple:
        pass

