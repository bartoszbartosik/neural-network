from typing import Callable

import numpy as np

from neuralnetwork.neuron import Neuron


class Layer:

    def __init__(self, input_array: np.ndarray, neurons_number: int, activation_function: Callable):
        self.input_array = input_array
        self.neurons = np.array([Neuron(input_array, activation_function) for _ in range(neurons_number)])
        self.size = len(self.neurons)
        self.activation_function = activation_function

    def array(self):
        return np.array([neuron.value for neuron in self.neurons])

    def feedforward(self):
        for neuron in self.neurons:
            neuron.input_array = self.input_array
            neuron.compute_value()

    def get_weights(self):
        return np.array([neuron.weights for neuron in self.neurons])

    def get_biases(self):
        return np.array([neuron.bias for neuron in self.neurons])

    def get_inputs(self):
        return np.array([neuron.input_value for neuron in self.neurons])


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # #   O V E R W R I T T E N   F U N C T I O N S   # # # # # # # # # # # # # # # # #
    def __str__(self) -> str:
        return 'Layer: [Neurons: {}]'.format([str(neuron.value) for neuron in self.neurons])

    def __repr__(self):
        return self.__str__()

