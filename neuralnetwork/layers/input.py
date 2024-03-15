from typing import Callable

import numpy as np

from neuralnetwork.layers import Layer
from neuralnetwork.activations import *


class InputLayer(Layer):
    def __init__(self, input_shape: tuple):
        super().__init__(neurons=np.prod([i for i in input_shape]),
                         activation=linear)

