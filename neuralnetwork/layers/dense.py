from typing import Callable

import numpy as np

from neuralnetwork.layers import Layer


class Dense(Layer):
    def __init__(self, neurons: int, activation: Callable, weights: np.ndarray = None, bias: float = None):
        super().__init__(neurons=neurons,
                         activation=activation,
                         weights=weights,
                         bias=bias)
