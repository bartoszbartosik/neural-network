from typing import Callable

import numpy as np

from neuralnetwork.layers import Layer
from neuralnetwork import losses
from neuralnetwork import activations


class Dense(Layer):

    def __init__(self, neurons: int, activation: Callable, weights: np.ndarray = None, bias: float = None):
        super().__init__(activation=activation,
                         weights=weights,
                         bias=bias)
        self.n = neurons
        self.z: np.ndarray = np.zeros(self.n)
        self.a: np.ndarray = np.zeros(self.n)


    def compile(self, a_: np.ndarray, loss: Callable) -> None:
        self.loss = loss
        if self.w is None:
            self.w = np.random.rand(self.n, len(a_)) * 2 - 1
        if self.b is None:
            self.b = np.random.random() * 2 - 1


    def feedforward(self, a_):
        self.z = (np.dot(self.w, a_.transpose()) + self.b).transpose()
        self.a = self.activation(self.z)


    def backpropagate(self,
                      l_prev: Layer,
                      delta_prev: np.ndarray = None,
                      l_next: Layer = None,
                      y: np.ndarray = None) -> tuple:
        """
        Calculate the Cost Function derivative over each weight and bias in order to determine its gradient.
        """
        # Output layer
        if y is not None:
            grad_b = losses.d(self.loss, self.a, y) * activations.d(self.activation, self.z)
        # Hidden layer
        else:
            grad_b = np.dot(delta_prev, l_next.w) * activations.d(self.activation, self.z)

        grad_w = [np.outer(d, a) for d, a in zip(grad_b, l_prev.a)]

        return grad_b, grad_w

