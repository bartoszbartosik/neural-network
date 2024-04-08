from typing import Callable

import numpy as np

from neuralnetwork.layers import Layer
from neuralnetwork.activations import linear


class MaxPooling(Layer):

    def __init__(self, pool_size: tuple, stride: int):
        super().__init__(activation=linear)
        self.pool_size = pool_size
        self.stride = stride
        self.amax = np.zeros([])
        self.indices = np.empty([], dtype=object)



    def build(self, a_: np.ndarray, loss: Callable) -> None:
        # Compute activations shape after pooling
        batch_size, rows, cols, channels = a_.shape
        p_rows, p_cols = self.pool_size
        out_rows = (rows - p_rows) // self.stride + 1
        out_cols = (cols - p_cols) // self.stride + 1
        self.shape = (batch_size, out_rows, out_cols, channels)
        self.a = np.zeros(self.shape)


    def feedforward(self, a_: np.ndarray) -> None:
        """
        Perform MaxPooling and MaxUnpooling to store gradients.
        """
        batch_size, out_rows, out_cols, channels = self.shape
        p_rows, p_cols = self.pool_size

        self.indices = np.empty(self.shape, dtype=object)
        for b in range(batch_size):
            for c in range(channels):
                for row in range(out_rows):
                    for col in range(out_cols):
                        # Slice appropriate window
                        window = a_[b, row*self.stride : row*self.stride + p_rows, col*self.stride : col*self.stride + p_cols, c]
                        # Get the max value
                        window_max = np.max(window)
                        # Update MaxPool activations
                        self.a[b, row, col, c] = window_max

                        # Get the max value's 2D index
                        r_max, c_max = np.unravel_index(np.argmax(window), window.shape)
                        r_max += row * self.stride
                        c_max += col * self.stride

                        # Store indices
                        self.indices[b, row, col, c] = r_max, c_max
                        # self.indices.append((r_max, c_max))


    def backpropagate(self, grad: np.ndarray, a_: Layer, w_out: Layer = None) -> tuple:
        """
        The gradients are the MaxUnpooled values of the MaxPooling layer activations.
        """
        batch_size, rows, cols, channels = self.shape

        grad_b = np.zeros_like(a_)
        grad = grad.reshape(self.shape)

        for b in range(batch_size):
            for c in range(channels):
                for row in range(len(self.indices[b, :, :, c])):
                    for col in range(len(self.indices[b, row, :, c][0])):
                        grad_b[b, :, :, c][self.indices[b, row, col, c]] = grad[b, row, col, c]

        return grad_b, None

