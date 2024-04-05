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


    def backpropagate(self, grad: np.ndarray, lin: Layer, lout: Layer = None) -> tuple:
        return grad.reshape(lin.shape), None

