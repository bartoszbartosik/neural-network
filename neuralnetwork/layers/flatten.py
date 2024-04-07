from typing import Callable

import numpy as np

from neuralnetwork.layers import Layer, Dense
from neuralnetwork.activations import linear
from neuralnetwork import activations


class Flatten(Layer):

    def __init__(self):
        super().__init__(activation=linear)
        self.input_shape = ()


    def build(self, a_: np.ndarray, loss: Callable) -> None:
        batch_size, rows, cols, channels = self.input_shape = a_.shape
        self.shape = (batch_size, rows*cols*channels)
        self.a = np.zeros(self.shape)


    def feedforward(self, a_: np.ndarray) -> None:
        batch_size, _, _, _ = a_.shape
        self.a = np.reshape(a_, (batch_size, -1))


    def backpropagate(self, grad: np.ndarray, a_: np.ndarray, w_out: np.ndarray = None) -> tuple:
        # If the layer is hidden, use weights from the output to scale the gradient
        grad = grad if w_out is None else np.dot(grad, w_out)

        return grad, None

