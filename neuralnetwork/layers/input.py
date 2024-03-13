from typing import Callable

import numpy as np

from neuralnetwork.layers.layer import Layer


class InputLayer(Layer):
    def __init__(self, input_array: np.ndarray, neurons_number: int, activation_function: Callable):
        super.__init__()
        super().__init__(input_array, neurons_number, activation_function)