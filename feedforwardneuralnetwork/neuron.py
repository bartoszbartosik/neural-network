from typing import Callable

import numpy as np


class Neuron:

    def __init__(self, input_array: np.ndarray, activation_function: Callable):
        # INITIALIZE INPUT LAYER
        self.input_array = input_array

        # INITIALIZE WEIGHTS
        # Weights number
        weights_number = len(input_array)

        # Weights array
        self.weights = np.random.uniform(low=-1, high=1, size=weights_number)

        # INITIALIZE BIAS
        self.bias = np.random.uniform(low=-1, high=1)

        # DEFINE ACTIVATION FUNCTION
        self.activation_function = activation_function

        # COMPUTE NEURON VALUE
        self.input_value = 0
        self.value = 0
        self.compute_value()


    def compute_value(self):
        self.input_value = np.dot(self.input_array, self.weights) + self.bias
        self.value = self.activation_function(self.input_value)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # #   P R I V A T E   F U N C T I O N S   # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # #   O V E R W R I T T E N   F U N C T I O N S   # # # # # # # # # # # # # # # # #
    def __str__(self) -> str:
        return 'Neuron: [value: {}, weights: {}, bias: {}]'.format(self.value, self.weights, self.bias)

    def __repr__(self):
        return self.__str__()
