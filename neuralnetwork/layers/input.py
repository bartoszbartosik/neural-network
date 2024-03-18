from typing import Callable

import numpy as np

from neuralnetwork.layers import Layer
from neuralnetwork.activations import *


class InputLayer(Layer):

    def __init__(self, input_shape: tuple):
        super().__init__(activation=linear)

        self.a = np.zeros(input_shape)


    def compile(self, a_: np.ndarray, loss: Callable) -> None:
        self.a = a_

    def feedforward(self, a_):
        self.a = a_

    def backpropagate(self, y):
        pass
