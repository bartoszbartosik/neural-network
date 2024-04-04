from typing import Callable

import numpy as np

from neuralnetwork.layers import Layer


def convolve(a: np.ndarray, kernel: np.ndarray, padding) -> np.ndarray:
    kernel = np.rot90(kernel, 2)
    return crosscorrelate(a, kernel, padding)


def crosscorrelate(a: np.ndarray, kernel: np.ndarray, padding: str) -> np.ndarray:
    # Unpack kernel shape
    krows, kcols = kernel.shape

    # Add padding if applicable
    if padding == 'same' or padding == 'full':
        pad_row = (krows//2, krows//2 - (krows % 2 == 0))
        pad_col = (kcols//2, kcols//2 - (kcols % 2 == 0))
        a = np.pad(a, pad_width=(pad_row, pad_col))

    # Get matrix shape
    rows, cols = a.shape

    # Compute output shape
    out_rows, out_cols = (rows - krows + 1, cols - kcols + 1)

    # Initialize result array
    conv = np.zeros((out_rows, out_cols))

    # Cross-correlate
    for row in range(out_rows):
        for col in range(out_cols):
            conv[row, col] = np.sum(a[row : row + krows, col : col + kcols] * kernel)

    return conv


class Convolutional(Layer):

    def __init__(self, kernels: int, kernel_size: tuple, activation: Callable, padding='valid'):
        super().__init__(activation=activation,
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
                    self.z[b, :, :, k] += crosscorrelate(a_[b, :, :, c], self.kernels[k, c, :, :], padding=self.padding) + self.biases[:, :, k]

        self.a = self.activation(self.z)


    def backpropagate(self,
                      grad: np.ndarray,
                      lin: Layer,
                      lout: Layer = None) -> tuple:

        batch_size, out_rows, out_cols, out_channels = lin.shape

        grad_b = np.zeros(self.shape)
        grad_w = np.zeros(self.kernels.shape)

        if lout is not None:
            for b in range(batch_size):
                for k in range(self.knum):
                    for c in range(out_channels):
                        grad_b[b, :, :, k] += convolve(grad[k, c, :, :], self.kernels[k, c, :, :], 'same')
                        grad_w[b, :, :, k] += crosscorrelate(lin.a[b, :, :, c], grad[k, c, :, :], 'valid')

        return grad_b, grad_w



